"""SOFTWARE_TEST_DOUBLE only: tiny fabricated bodies, no EEG/network/GPU/fit."""
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from tools import source_preparation_batch as base
from tools import source_preparation_parallel as parallel
from tests.test_source_preparation_batch import DATA, fake_native, response


def budget(path, plan=None, **kwargs):
    return parallel.LockedResourceBudget(base.ResourceBudget(plan or base.build_plan(max_download_bytes=100), path, **kwargs))


def test_two_scopes_same_sequence_are_independent_and_snapshot_is_detached(tmp_path):
    b = budget(tmp_path / "resource.json")
    try:
        official = b.reserve_transfer(7, object_id="SOFTWARE_PROVIDER", probe_bytes=1)
        r2 = b.reserve_transfer(5, object_id="SOFTWARE_R2", probe_bytes=3, scope=base.ResourceBudget.R2_BACKUP)
        assert official == r2 == 1
        saved = json.loads(b.path.read_text())
        assert sum(a["status"] == parallel.OPEN for ledger in saved["byte_scopes"].values() for a in ledger["attempts"]) == 2
        with pytest.raises(base.BatchError, match="limit_two"):
            b.reserve_transfer(1, object_id="third")
        copy = b.data
        copy["byte_scopes"][b.scope]["charged_bytes"] = 0
        assert b.snapshot()["byte_scopes"][b.scope]["charged_bytes"] == 8
        b.consume(7, transfer_id=official); b.finish_transfer(official, complete_body_read=True)
        b.consume(5, transfer_id=r2, scope=base.ResourceBudget.R2_BACKUP)
        b.finish_transfer(r2, complete_body_read=True, scope=base.ResourceBudget.R2_BACKUP)
        assert b.data["byte_scopes"][b.scope]["charged_bytes"] == 7
        assert b.data["byte_scopes"][base.ResourceBudget.R2_BACKUP]["charged_bytes"] == 5
        with pytest.raises(base.BatchError, match="matching_open"):
            b.finish_transfer(official, complete_body_read=True)
    finally:
        b.close()


def test_concurrent_reservation_cannot_oversubscribe_single_original_cap(tmp_path):
    b = budget(tmp_path / "resource.json", base.build_plan(max_download_bytes=10))
    barrier = threading.Barrier(2)
    def reserve(i):
        barrier.wait(5)
        try:
            return b.reserve_transfer(6, object_id=f"SOFTWARE_{i}", probe_bytes=1)
        except base.BatchError:
            return None
    try:
        with ThreadPoolExecutor(2) as pool:
            tokens = list(pool.map(reserve, range(2)))
        assert sum(t is not None for t in tokens) == 1
        assert b.data["download_bytes_charged"] == 7
    finally:
        b.close()


@pytest.mark.parametrize("complete,count,charged", [(False, 4, 8), (True, 4, 8), (True, 7, 7)])
def test_refund_only_exact_full_body_once(tmp_path, complete, count, charged):
    b = budget(tmp_path / "resource.json")
    token = b.reserve_transfer(7, object_id="SOFTWARE", probe_bytes=1)
    b.consume(count, transfer_id=token)
    b.finish_transfer(token, complete_body_read=complete)
    assert b.data["download_bytes_charged"] == charged
    with pytest.raises(base.BatchError):
        b.consume(1, transfer_id=token)
    with pytest.raises(base.BatchError):
        b.finish_transfer(token, complete_body_read=True)
    b.close()


def test_overread_is_durable_before_cap_error_and_no_refund(tmp_path):
    b = budget(tmp_path / "resource.json", base.build_plan(max_download_bytes=5))
    token = b.reserve_transfer(5, object_id="SOFTWARE")
    with pytest.raises(base.BatchError, match="byte_budget"):
        b.consume(6, transfer_id=token)
    saved = json.loads(b.path.read_text())
    assert saved["download_bytes_read"] == saved["download_bytes_charged"] == 6
    b.finish_transfer(token, complete_body_read=True)
    assert b.data["download_bytes_charged"] == 6
    b.close()


def test_reopen_keeps_parallel_and_legacy_unfinished_reservations_fully_charged(tmp_path):
    plan = base.build_plan(max_download_bytes=100)
    path = tmp_path / "resource.json"
    b = budget(path, plan)
    token = b.reserve_transfer(7, object_id="SOFTWARE_OPEN", probe_bytes=1)
    b.consume(2, transfer_id=token); b.close()
    b = budget(path, plan)
    assert b.data["download_bytes_charged"] == 8 and b.data["download_bytes_read"] == 2
    assert b.data["byte_scopes"][b.scope]["attempts"][0]["status"] == parallel.UNCLEAN
    with pytest.raises(base.BatchError):
        b.finish_transfer(token, complete_body_read=True)
    b.close()
    legacy_path = tmp_path / "legacy.json"
    old = base.ResourceBudget(plan, legacy_path)
    old.reserve_transfer(7, object_id="SOFTWARE_LEGACY", probe_bytes=1); old.close()
    b = budget(legacy_path, plan)
    assert b.data["download_bytes_charged"] == 8
    assert b.data["byte_scopes"][b.scope]["active"] is None
    assert b.data["byte_scopes"][b.scope]["attempts"][0]["status"] == "UNCLEAN_FULL_RESERVATION_REMAINS_CHARGED"
    b.close()


def test_actual_self_child_SIGKILL_two_open_software_reservations_survive(tmp_path):
    path = tmp_path / "resource.json"
    script = """
import os,signal,sys
from pathlib import Path
from tools.source_preparation_batch import ResourceBudget,build_plan
from tools.source_preparation_parallel import LockedResourceBudget
b=LockedResourceBudget(ResourceBudget(build_plan(max_download_bytes=100),Path(sys.argv[1])))
t=b.reserve_transfer(7,object_id='SOFTWARE_PROVIDER',probe_bytes=1)
b.consume(2,transfer_id=t)
t=b.reserve_transfer(9,object_id='SOFTWARE_R2',scope=ResourceBudget.R2_BACKUP,probe_bytes=3)
b.consume(3,transfer_id=t,scope=ResourceBudget.R2_BACKUP)
os.kill(os.getpid(),signal.SIGKILL)
"""
    result = subprocess.run([sys.executable, "-c", script, str(path)], cwd=parallel.REPO,
                            capture_output=True, timeout=20)
    assert result.returncode == -signal.SIGKILL
    b = budget(path)
    try:
        assert b.data["download_bytes_charged"] == 8 and b.data["download_bytes_read"] == 2
        assert b.data["byte_scopes"][base.ResourceBudget.R2_BACKUP]["charged_bytes"] == 12
        assert all(a["status"] == parallel.UNCLEAN for row in b.data["byte_scopes"].values() for a in row["attempts"])
    finally:
        b.close()


@pytest.mark.parametrize("failure", [ValueError, MemoryError])
def test_native_audit_lock_releases_traceback_args_attributes_and_arrays(tmp_path, monkeypatch, failure):
    path = tmp_path / "SOFTWARE_NOT_EEG.mat"; path.write_bytes(b"not EEG")
    references = []
    def native(*args, **kwargs):
        array = np.ones((20, 20))
        references.append(weakref.ref(array))
        error = failure(array)
        error.extra_array = array
        raise error
    monkeypatch.setattr(parallel, "resource_preflight", lambda *a: {})
    monkeypatch.setattr(base, "audit_native_source", native)
    audit = parallel.SingleNativeAudit(tmp_path)
    with pytest.raises(failure) as error:
        audit("Cho2017", path)
    gc.collect()
    assert references[0]() is None
    assert error.value.args == () and error.value.__dict__ == {}
    monkeypatch.setattr(base, "audit_native_source", lambda *a, **kw: {"trials": [], "summary": {"SOFTWARE": True}})
    assert audit("Cho2017", path)["summary"]["SOFTWARE"]


def test_native_success_must_not_return_large_opaque_array(tmp_path, monkeypatch):
    path = tmp_path / "SOFTWARE_NOT_EEG.mat"; path.write_bytes(b"not EEG")
    references = []
    def native(*args, **kwargs):
        array = np.ones((20, 20)); references.append(weakref.ref(array))
        return {"trials": [], "opaque_EEG": array}
    monkeypatch.setattr(parallel, "resource_preflight", lambda *a: {})
    monkeypatch.setattr(base, "audit_native_source", native)
    with pytest.raises(TypeError):
        parallel.SingleNativeAudit(tmp_path)("Cho2017", path)
    gc.collect(); assert references[0]() is None


class SoftwareSDK:
    """Local dictionaries and BytesIO, not an actual R2 client or receipt proof."""
    def __init__(self, resource_path):
        self.objects, self.lock, self.get_count, self.resource_path = {}, threading.Lock(), 0, resource_path
        self.close_count = 0

    def close(self):
        self.close_count += 1

    def put_object(self, **kwargs):
        blob = kwargs["Body"].read()
        with self.lock:
            self.objects[kwargs["Key"]] = blob
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def get_object(self, **kwargs):
        saved = json.loads(self.resource_path.read_text())
        scope = saved["byte_scopes"][base.ResourceBudget.R2_BACKUP]
        assert any(a["status"] == parallel.OPEN and a["reserved_bytes"] > 0 for a in scope["attempts"])
        with self.lock:
            self.get_count += 1
            blob = self.objects[kwargs["Key"]]
        return {"ResponseMetadata": {"HTTPStatusCode": 200}, "ContentLength": len(blob), "Body": io.BytesIO(blob)}


@pytest.fixture
def software_executor(tmp_path, monkeypatch):
    plan = base.build_plan()
    base.atomic_json(tmp_path / "status/tasks.json", {"W1": {"status": "COMPLETE"}, "W2": {"status": "PENDING"}})
    base.atomic_json(tmp_path / "audit/source_schema/evidence_manifest.json", {"files": [], "SOFTWARE_TEST_DOUBLE": True})
    monkeypatch.setattr(base, "execution_gate", lambda root: "a" * 64)
    monkeypatch.setattr(base, "check_space", lambda *a: None)
    monkeypatch.setattr(parallel.shutil, "disk_usage", lambda *a: SimpleNamespace(free=1024**4))
    monkeypatch.setattr(parallel, "resource_preflight", lambda *a: {"SOFTWARE_TEST_DOUBLE": True})
    state = SimpleNamespace(parser_active=0, parser_peak=0, io_active=0, io_peak=0, head_ids=[], lock=threading.Lock(),
        barrier=threading.Barrier(2), first_reads=0, audit_ids=[])
    class Body(io.BytesIO):
        entered = False
        def read(self, n=-1):
            if not self.entered:
                self.entered = True
                with state.lock:
                    state.io_active += 1; state.io_peak = max(state.io_peak, state.io_active)
                    state.first_reads += 1; wait_for_peer = state.first_reads <= 2
                if wait_for_peer: state.barrier.wait(5)
            result = super().read(n)
            if not result and self.entered:
                with state.lock: state.io_active -= 1
                self.entered = False
            return result
    def opener(request, **kwargs):
        if request.get_method() == "HEAD":
            with state.lock: state.head_ids.append(request.full_url)
            return response(size=len(DATA))
        obj = Body(DATA)
        obj.status, obj.headers = 200, response(size=len(DATA)).headers
        return obj
    monkeypatch.setattr(base.urllib.request, "urlopen", opener)
    def native(dataset, path, **kwargs):
        with state.lock:
            state.parser_active += 1; state.parser_peak = max(state.parser_peak, state.parser_active)
            state.audit_ids.append(path.name)
        try:
            time.sleep(.005)
            result = fake_native(dataset, path, **kwargs)
            result["trials"][0]["trial_uid"] = hashlib.sha256((path.name + kwargs["expected_sha256"]).encode()).hexdigest()
            return result
        finally:
            with state.lock: state.parser_active -= 1
    monkeypatch.setattr(base, "audit_native_source", native)
    sdk = SoftwareSDK(tmp_path / "audit/source_preparation/resource_progress.json")
    return SimpleNamespace(root=tmp_path, plan=plan, state=state, sdk=sdk, native=native)


def run_fixture(f, **kwargs):
    return parallel.execute_plan(f.plan, root=f.root, client=f.sdk, bucket="SOFTWARE_NOT_REAL_BUCKET",
                                 max_new_files=kwargs.pop("max_new_files", 2), **kwargs)


def test_two_body_streams_one_parser_native_instrumentation_and_unchanged_plan(software_executor, monkeypatch):
    f = software_executor
    monkeypatch.setattr(base, "create_client", lambda: pytest.fail("shared low-level client was ignored"))
    before = json.dumps(f.plan, sort_keys=True)
    result = run_fixture(f)
    assert result["actual_verified_objects"] == 2 and result["inventory_status"] == "SOURCE_RAW_INVENTORY_PARTIAL"
    assert f.state.io_peak == 2 and f.state.parser_peak == 1
    assert f.sdk.get_count == 6 and json.dumps(f.plan, sort_keys=True) == before
    assert result["download_bytes_read"] == result["download_bytes_charged"] == 2 * len(DATA)
    assert f.sdk.close_count == 0  # Injected SDK remains caller-owned.
    r2 = result["byte_scopes"][base.ResourceBudget.R2_BACKUP]
    assert r2["observed_bytes"] == r2["charged_bytes"] > 2 * len(DATA)
    assert json.loads((f.root / "status/tasks.json").read_text())["W2"]["status"] == "PENDING"
    amendment = json.loads((f.root / result["operational_amendment"]["path"]).read_text())
    assert amendment["base_plan_sha256"] == f.plan["plan_sha256"]
    assert amendment["inflight_file_lifecycles"] == 2 and amendment["parallel_native_parsers"] == 1
    assert f.plan["inflight_downloads"] == f.plan["parallel_native_parsers"] == 1
    assert not list((f.root / "data/source_preparation/staging").glob("*/*.mat"))
    # Completed replay must not even HEAD, parse or GET. Reuse the original
    # identity/journal instead of fabricating a new accounting namespace.
    path = f.root / result["resource_progress_path"]
    saved = json.loads(path.read_text())
    b = budget(path, f.plan, identity_sha256=saved["resource_identity_sha256"])
    def forbidden(*a, **kw): pytest.fail("Completed replay attempted HEAD/GET/native parser")
    count = f.sdk.get_count
    try:
        for entry in f.plan["files"][:2]:
            state = base.process_file(entry, root=f.root, batch_dir=f.root / "audit/source_preparation/batches" / f.plan["plan_sha256"],
                plan=f.plan, contract_sha256="a" * 64, budget=b, audit_fn=forbidden, opener=forbidden,
                client=f.sdk, bucket="SOFTWARE_NOT_REAL_BUCKET")
            assert state["status"] == "COMPLETE"
    finally:
        b.close()
    assert f.sdk.get_count == count


def test_resource_pause_is_retriable_not_a_native_schema_judgment(software_executor, monkeypatch):
    f = software_executor
    monkeypatch.setattr(parallel, "resource_preflight", lambda *a: (_ for _ in ()).throw(RuntimeError("SOFTWARE_RESOURCE_PAUSE")))
    result = run_fixture(f)
    assert result["inventory_status"] == "SOURCE_RAW_INVENTORY_PARTIAL" and f.sdk.get_count == 0
    states = list((f.root / "audit/source_preparation/batches" / f.plan["plan_sha256"] / "objects").glob("*/state.json"))
    assert len(states) == 2 and all(json.loads(p.read_text())["status"] == "RAW_VERIFIED" for p in states)
    assert len(list((f.root / "data/source_preparation/staging").glob("*/*.mat"))) == 2


def test_archived_schema_unknown_can_finish_independent_objects_but_not_full_inventory(software_executor, monkeypatch):
    f = software_executor
    monkeypatch.setattr(base, "audit_native_source", lambda *a, **kw: (_ for _ in ()).throw(ValueError("SOFTWARE_NATIVE_UNKNOWN")))
    result = run_fixture(f)
    assert result["schema_blocked"] == 2 and result["actual_verified_objects"] == 0
    assert result["inventory_status"] != "SOURCE_RAW_INVENTORY_COMPLETE" and f.sdk.get_count == 6
    assert not list((f.root / "data/source_preparation/staging").glob("*/*.mat"))


@pytest.mark.parametrize("missing", ["byte_only", "wall_only", "state_without_pair", "partial_without_pair"])
def test_missing_existing_accounting_cannot_bootstrap_a_zero_budget(software_executor, missing, monkeypatch):
    f = software_executor
    path = f.root / "audit/source_preparation/resource_progress.json"
    if missing == "byte_only": base.atomic_json(path, {"SOFTWARE_TEST_DOUBLE": True})
    elif missing == "wall_only": base.atomic_json(path.with_name("resource_progress.wall.json"), {"SOFTWARE_TEST_DOUBLE": True})
    elif missing == "state_without_pair":
        base.atomic_json(f.root / "audit/source_preparation/batches" / f.plan["plan_sha256"] / "objects/x/state.json", {"status": "RAW_VERIFIED"})
    else:
        raw = base.raw_staging_path(f.root, f.plan, f.plan["files"][0]).with_suffix(".mat.partial")
        raw.parent.mkdir(parents=True); raw.write_bytes(b"SOFTWARE_PARTIAL")
    monkeypatch.setattr(base, "create_client", lambda: pytest.fail("SDK created before surviving-accounting gate"))
    with pytest.raises(base.BatchError, match="journals"):
        run_fixture(f)
    assert not f.state.head_ids


def test_empty_proposal_directories_are_not_misclassified_as_a_prior_transfer(tmp_path):
    plan = base.build_plan()
    folder = tmp_path / "audit/source_preparation/batches" / plan["plan_sha256"]
    base.atomic_json(folder / "parallel_operational_amendment_proposal.json", {"SOFTWARE_PROPOSAL_ONLY": True})
    (tmp_path / "data/source_schema/Cho2017").mkdir(parents=True)
    (tmp_path / "data/source_schema/Cho2017/preserved-schema.mat").write_bytes(b"SOFTWARE_PRESERVED_SAMPLE")
    parallel._require_accounting_pair(tmp_path, folder)


def stage_fixture(f, positions):
    folder = f.root / "audit/source_preparation/batches" / f.plan["plan_sha256"]
    for position in positions:
        entry = f.plan["files"][position]
        raw = base.raw_staging_path(f.root, f.plan, entry)
        raw.parent.mkdir(parents=True); raw.write_bytes(DATA)
        base.atomic_json(folder / "objects" / entry["object_id"] / "state.json", {
            "identity": list(base.identity(entry)), "plan_sha256": f.plan["plan_sha256"],
            "temporary_raw_owned": True, "raw_path": raw.relative_to(f.root).as_posix(),
            "input_contract_sha256": "a" * 64, "status": "RAW_VERIFIED"})
    return folder


def test_existing_owned_file_late_in_roster_is_recovered_before_new_staging(software_executor):
    f = software_executor
    folder = stage_fixture(f, [100])
    ordered = parallel._staged_priority(f.root, f.plan, folder)
    assert ordered[0] == f.plan["files"][100]
    assert {r["object_id"] for r in ordered} == {r["object_id"] for r in f.plan["files"]}


def test_executor_actually_recovers_late_owned_raw_without_new_provider_GET(software_executor, monkeypatch):
    f = software_executor
    path = f.root / "audit/source_preparation/resource_progress.json"
    identity = base.digest({"schema": "source-raw-resource-identity-v2", "files": f.plan["files"],
        "adapter_code_sha256": f.plan["adapter_code_sha256"], "input_contract_sha256": "a" * 64})
    prior = budget(path, f.plan, identity_sha256=identity)
    token = prior.reserve_transfer(len(DATA), object_id="SOFTWARE_OLD_BODY", probe_bytes=1)
    prior.consume(len(DATA), transfer_id=token); prior.finish_transfer(token, complete_body_read=True); prior.close()
    stage_fixture(f, [100])
    def only_HEAD(request, **kwargs):
        assert request.get_method() == "HEAD", "retained verified raw must not be fetched again"
        f.state.head_ids.append(request.full_url)
        return response(size=len(DATA))
    monkeypatch.setattr(base.urllib.request, "urlopen", only_HEAD)
    result = run_fixture(f, max_new_files=1)
    assert result["actual_verified_objects"] == 1 and f.sdk.get_count == 3
    assert f.state.head_ids == [f.plan["files"][100]["url"]]
    assert result["download_bytes_read"] == len(DATA)  # Only prior charged history.
    assert not list((f.root / "data/source_preparation/staging").glob("*/*.mat"))


def test_more_than_two_retained_staging_lifecycles_or_unowned_raw_are_preserved(software_executor):
    f = software_executor
    folder = stage_fixture(f, [20, 50, 100])
    with pytest.raises(base.BatchError, match="more_than_two"):
        parallel._staged_priority(f.root, f.plan, folder)
    assert len(list((f.root / "data/source_preparation/staging").glob("*/*.mat"))) == 3
    state = folder / "objects" / f.plan["files"][20]["object_id"] / "state.json"
    value = json.loads(state.read_text()); value["temporary_raw_owned"] = False; base.atomic_json(state, value)
    with pytest.raises(base.BatchError, match="unowned"):
        parallel._staged_priority(f.root, f.plan, folder)


@pytest.mark.parametrize("unknown", ["directory", "file", "unexpected_file_in_known_directory"])
def test_unknown_staging_bytes_are_preserved_and_block_new_slots(software_executor, unknown):
    f = software_executor
    folder = f.root / "audit/source_preparation/batches" / f.plan["plan_sha256"]
    if unknown == "unexpected_file_in_known_directory":
        path = base.raw_staging_path(f.root, f.plan, f.plan["files"][0]).parent / "orphan-not-original.partial"
    elif unknown == "directory":
        path = f.root / "data/source_preparation/staging/unknown-id/unknown.mat"
    else:
        path = f.root / "data/source_preparation/staging/unexpected.mat"
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"SOFTWARE_UNKNOWN_PRESERVED")
    with pytest.raises(base.BatchError, match="unknown_source_staging"):
        parallel._staged_priority(f.root, f.plan, folder)
    assert path.read_bytes() == b"SOFTWARE_UNKNOWN_PRESERVED"


def test_corrupt_completed_receipt_in_retained_late_slot_never_dispatches_new_files(software_executor):
    f = software_executor
    path = f.root / "audit/source_preparation/resource_progress.json"
    identity = base.digest({"schema": "source-raw-resource-identity-v2", "files": f.plan["files"],
        "adapter_code_sha256": f.plan["adapter_code_sha256"], "input_contract_sha256": "a" * 64})
    prior = budget(path, f.plan, identity_sha256=identity); prior.close()
    folder = stage_fixture(f, [100])
    state_path = folder / "objects" / f.plan["files"][100]["object_id"] / "state.json"
    value = json.loads(state_path.read_text())
    value.update(status="COMPLETE", retained_metadata_sha256={"audit/does-not-exist.json": "f" * 64})
    base.atomic_json(state_path, value)
    result = run_fixture(f)
    assert result["inventory_status"] == "SOURCE_RAW_INVENTORY_PARTIAL" and result["failed"] == 1
    assert not f.state.head_ids and f.sdk.get_count == 0
    assert len(list((f.root / "data/source_preparation/staging").glob("*/*.mat"))) == 1


def test_implicitly_created_shared_SDK_is_closed_once_after_draining(software_executor, monkeypatch):
    f = software_executor
    monkeypatch.setenv("R2_BUCKET", "SOFTWARE_NOT_REAL_BUCKET")
    created_threads = []
    def create():
        created_threads.append(threading.current_thread())
        return f.sdk
    monkeypatch.setattr(base, "create_client", create)
    result = parallel.execute_plan(f.plan, root=f.root, max_new_files=2)
    assert result["actual_verified_objects"] == 2
    assert created_threads == [threading.main_thread()] and f.sdk.close_count == 1


def test_default_CLI_is_read_only_and_W1_blocks_before_SDK_or_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "create_client", lambda: pytest.fail("SDK from dryrun"))
    assert parallel.main(["--root", str(tmp_path)]) == 0
    assert list(tmp_path.iterdir()) == []
    base.atomic_json(tmp_path / "status/tasks.json", {"W1": {"status": "PENDING"}})
    with pytest.raises(base.BatchError, match="w1_complete"):
        parallel.execute_plan(base.build_plan(), root=tmp_path)
    assert not (tmp_path / "audit").exists()


def test_implicit_SDK_creation_from_worker_thread_is_rejected_before_any_IO(software_executor):
    f = software_executor
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(parallel.execute_plan, f.plan, root=f.root,
                                 bucket="SOFTWARE_NOT_REAL_BUCKET", max_new_files=1)
        with pytest.raises(base.BatchError, match="main_thread"):
            future.result()
    assert not f.state.head_ids and f.sdk.get_count == 0
    assert not f.sdk.resource_path.exists()


def test_CLI_keeps_full_original_cache_R2_pipeline_and_fixed_coordinator_arguments(tmp_path, monkeypatch):
    from tools import prepare_all_sources as pipeline
    calls = []
    def prepare(root, **kwargs):
        calls.append((root, kwargs))
        return {"status": "W2_COMPLETE_CACHE_AND_R2_VERIFIED", "SOFTWARE_TEST_DOUBLE_NOT_REAL_PREPARATION": True}
    monkeypatch.setattr(pipeline, "prepare_all", prepare)
    assert parallel.main(["--root", str(tmp_path), "--progress", "audit/future-source.json",
                          "--contract", "contracts/input_contract.json", "--execute"]) == 0
    root, options = calls[0]
    assert root == tmp_path and options["raw_fn"] is parallel.execute_plan
    assert options["contract"] == "contracts/input_contract.json" and options["execute"] is True
    assert options["progress"] == "audit/future-source.json"
