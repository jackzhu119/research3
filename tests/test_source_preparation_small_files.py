"""SOFTWARE_TEST_DOUBLE only: fabricated metadata/bodies, no network/EEG/GPU.

The scientific 505-object plan is retained. No test result is a source download,
native-schema finding, R2 backup, numerical cache or research completion claim.
"""
from concurrent.futures import ThreadPoolExecutor
import io
import json
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from tools import source_preparation_batch as base
from tools import source_preparation_parallel as parallel
from tools import source_preparation_small_files as small
from tests.test_source_preparation_batch import response


CONTRACT = "a" * 64
BODY = b"SOFTWARE_TEST_DOUBLE_NOT_EEG"


def resource(path, plan=None, *, max_open=16, **kwargs):
    raw = base.ResourceBudget(plan or base.build_plan(max_download_bytes=10000), path, **kwargs)
    try:
        return small.SmallFileResourceBudget(raw, max_open=max_open)
    except BaseException:
        raw.close()
        raise


@pytest.mark.parametrize("limit", [1, 2, 16])
def test_exact_lease_limit_across_scopes_with_detached_snapshot(tmp_path, limit):
    budget = resource(tmp_path / "resource.json", max_open=limit)
    try:
        leases = []
        for i in range(limit):
            scope = budget.scope if i % 2 == 0 else base.ResourceBudget.R2_BACKUP
            token = budget.reserve_transfer(7, object_id=f"SOFTWARE_{i}", probe_bytes=1, scope=scope)
            leases.append((scope, token))
        with pytest.raises(base.BatchError):
            budget.reserve_transfer(1, object_id="SOFTWARE_EXCESS")
        snapshot = budget.snapshot()
        assert sum(a["status"] == parallel.OPEN for row in snapshot["byte_scopes"].values() for a in row["attempts"]) == limit
        snapshot["byte_scopes"][budget.scope]["charged_bytes"] = 0
        assert budget.snapshot()["byte_scopes"][budget.scope]["charged_bytes"] >= 8
        for scope, token in leases:
            budget.consume(7, transfer_id=token, scope=scope)
            budget.finish_transfer(token, complete_body_read=True, scope=scope)
        assert sum(row["charged_bytes"] for row in budget.snapshot()["byte_scopes"].values()) == 7 * limit
    finally:
        budget.close()


@pytest.mark.parametrize("limit", [0, 17, -1, True, 1.5, "16"])
def test_invalid_lifecycle_limit_rejected(tmp_path, limit):
    with pytest.raises(base.BatchError):
        resource(tmp_path / "resource.json", max_open=limit)


def test_concurrent_reservations_preserve_one_original_byte_cap(tmp_path):
    budget = resource(tmp_path / "resource.json", base.build_plan(max_download_bytes=60))
    barrier = threading.Barrier(16)
    def reserve(index):
        barrier.wait(5)
        try:
            return budget.reserve_transfer(7, object_id=f"SOFTWARE_{index}", probe_bytes=1)
        except base.BatchError:
            return None
    try:
        with ThreadPoolExecutor(16) as executor:
            tokens = list(executor.map(reserve, range(16)))
        assert sum(token is not None for token in tokens) == 7
        assert budget.snapshot()["download_bytes_charged"] == 56
    finally:
        budget.close()


@pytest.mark.parametrize("complete,observed,charged", [(False, 4, 8), (True, 4, 8), (True, 7, 7)])
def test_only_exact_success_refunds_probe_once(tmp_path, complete, observed, charged):
    budget = resource(tmp_path / "resource.json")
    try:
        token = budget.reserve_transfer(7, object_id="SOFTWARE", probe_bytes=1)
        budget.consume(observed, transfer_id=token)
        budget.finish_transfer(token, complete_body_read=complete)
        assert budget.snapshot()["download_bytes_charged"] == charged
        with pytest.raises(base.BatchError):
            budget.finish_transfer(token, complete_body_read=True)
    finally:
        budget.close()


def test_overread_charge_is_persisted_before_cap_failure(tmp_path):
    budget = resource(tmp_path / "resource.json", base.build_plan(max_download_bytes=5))
    try:
        token = budget.reserve_transfer(5, object_id="SOFTWARE")
        with pytest.raises(base.BatchError):
            budget.consume(6, transfer_id=token)
        saved = json.loads(budget.path.read_text())
        assert saved["download_bytes_read"] == saved["download_bytes_charged"] == 6
        budget.finish_transfer(token, complete_body_read=True)
        assert budget.snapshot()["download_bytes_charged"] == 6
    finally:
        budget.close()


def test_self_SIGKILL_preserves_sixteen_full_reservations_without_refund(tmp_path):
    path = tmp_path / "resource.json"
    script = """
import os,signal,sys
from pathlib import Path
from tools.source_preparation_batch import ResourceBudget,build_plan
from tools.source_preparation_small_files import SmallFileResourceBudget
b=SmallFileResourceBudget(ResourceBudget(build_plan(max_download_bytes=10000),Path(sys.argv[1])),max_open=16)
for i in range(16):
 t=b.reserve_transfer(7,object_id='SOFTWARE_'+str(i),probe_bytes=1)
 b.consume(2,transfer_id=t)
os.kill(os.getpid(),signal.SIGKILL)
"""
    result = subprocess.run([sys.executable, "-c", script, str(path)], cwd=parallel.REPO,
        capture_output=True, timeout=20)
    assert result.returncode == -signal.SIGKILL
    budget = resource(path)
    try:
        saved = budget.snapshot()
        assert saved["download_bytes_charged"] == 128 and saved["download_bytes_read"] == 32
        assert len(saved["byte_scopes"][budget.scope]["attempts"]) == 16
        assert all(a["status"] == parallel.UNCLEAN for a in saved["byte_scopes"][budget.scope]["attempts"])
    finally:
        budget.close()


@pytest.fixture
def staging(tmp_path, monkeypatch):
    plan = base.build_plan()
    monkeypatch.setattr(base, "execution_gate", lambda root: CONTRACT)
    folder = tmp_path / "audit/source_preparation/batches" / plan["plan_sha256"]
    return SimpleNamespace(root=tmp_path, plan=plan, folder=folder,
        nonphysio=[row for row in plan["files"] if row["dataset"] != "PhysionetMI"],
        physio=[row for row in plan["files"] if row["dataset"] == "PhysionetMI"])


def stage(fixture, entry, *, partial=False):
    raw = base.raw_staging_path(fixture.root, fixture.plan, entry)
    if partial:
        raw = raw.with_name(raw.name + ".partial")
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(BODY)
    path = fixture.folder / "objects" / entry["object_id"] / "state.json"
    base.atomic_json(path, {"SOFTWARE_TEST_DOUBLE": True, "identity": list(base.identity(entry)),
        "plan_sha256": fixture.plan["plan_sha256"], "temporary_raw_owned": True,
        "raw_path": base.raw_staging_path(fixture.root, fixture.plan, entry).relative_to(fixture.root).as_posix(),
        "head_observation": {"http_status": 200, "bytes": len(BODY), "response_headers": response(size=len(BODY)).headers},
        "input_contract_sha256": CONTRACT, "status": "RAW_VERIFIED"})
    return raw, path


def test_staging_allows_one_large_and_fifteen_small_owned_lifecycles(staging):
    fixture = staging
    stage(fixture, fixture.nonphysio[-1], partial=True)
    for entry in fixture.physio[:15]:
        stage(fixture, entry, partial=True)
    nonphysio, physio = small.inspect_staging(fixture.root, fixture.plan, fixture.folder, contract_sha256=CONTRACT)
    assert nonphysio == [fixture.nonphysio[-1]]
    assert {row["object_id"] for row in physio} == {row["object_id"] for row in fixture.physio[:15]}


@pytest.mark.parametrize("kind", ["two_nonphysio", "seventeen_physio", "one_large_plus_sixteen_physio"])
def test_staging_limit_violation_preserves_every_original_byte(staging, kind):
    fixture = staging
    entries = (fixture.nonphysio[:2] if kind == "two_nonphysio" else fixture.physio[:17]
        if kind == "seventeen_physio" else fixture.nonphysio[:1]+fixture.physio[:16])
    paths = [stage(fixture, entry)[0] for entry in entries]
    with pytest.raises(base.BatchError):
        small.inspect_staging(fixture.root, fixture.plan, fixture.folder, contract_sha256=CONTRACT)
    assert all(path.read_bytes() == BODY for path in paths)


@pytest.mark.parametrize("kwargs", [{"max_file_workers": 17}, {"max_file_workers": 0},
    {"max_file_workers": True}, {"max_small_raw_bytes": 16*1024**2+1},
    {"max_small_raw_bytes": 0}, {"max_small_raw_bytes": True}])
def test_staging_operational_limits_cannot_exceed_fixed_typed_caps(staging, kwargs):
    fixture = staging
    with pytest.raises(base.BatchError):
        small.inspect_staging(fixture.root, fixture.plan, fixture.folder,
            contract_sha256=CONTRACT, **kwargs)


@pytest.mark.parametrize("field,value", [("temporary_raw_owned", False), ("plan_sha256", "b"*64),
    ("identity", ["OTHER"]), ("raw_path", "data/source_preparation/staging/other/file.edf"),
    ("input_contract_sha256", "b"*64)])
def test_staging_exact_ownership_identity_plan_and_contract_binding(staging, field, value):
    fixture = staging
    raw, state_path = stage(fixture, fixture.physio[0])
    state = json.loads(state_path.read_text()); state[field] = value; base.atomic_json(state_path, state)
    with pytest.raises(base.BatchError):
        small.inspect_staging(fixture.root, fixture.plan, fixture.folder, contract_sha256=CONTRACT)
    assert raw.read_bytes() == BODY


@pytest.mark.parametrize("kind", ["unknown_directory", "top_level_file", "wrong_basename", "symlink"])
def test_unknown_or_symlink_staging_blocks_without_cleanup(staging, kind):
    fixture = staging
    raw, _ = stage(fixture, fixture.physio[0])
    if kind == "wrong_basename":
        bad = raw.with_name("unowned.partial"); bad.write_bytes(BODY)
    elif kind == "unknown_directory":
        bad = fixture.root / "data/source_preparation/staging/unknown/file.edf"
        bad.parent.mkdir(); bad.write_bytes(BODY)
    elif kind == "top_level_file":
        bad = fixture.root / "data/source_preparation/staging/unknown.edf"; bad.write_bytes(BODY)
    else:
        bad = raw.with_name("unowned.partial"); bad.symlink_to(raw)
    with pytest.raises(base.BatchError):
        small.inspect_staging(fixture.root, fixture.plan, fixture.folder, contract_sha256=CONTRACT)
    assert raw.read_bytes() == BODY and bad.exists()


def test_default_CLI_is_read_only_and_never_opens_SDK(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "create_client", lambda: pytest.fail("dry run created SDK"))
    assert small.main(["--root", str(tmp_path)]) == 0
    assert list(tmp_path.iterdir()) == []


def test_W1_failure_precedes_SDK_or_new_staging(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "execution_gate", lambda root: (_ for _ in ()).throw(base.BatchError("SOFTWARE_W1_BLOCKED")))
    monkeypatch.setattr(base, "create_client", lambda: pytest.fail("SDK before W1 gate"))
    with pytest.raises(base.BatchError, match="SOFTWARE_W1_BLOCKED"):
        small.execute_plan(base.build_plan(), root=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("method", ["HEAD", "GET"])
def test_small_official_header_cap_is_checked_before_any_body_read(staging, monkeypatch, method):
    fixture = staging
    calls = []
    class ForbiddenBody(io.BytesIO):
        def read(self, *args):
            pytest.fail("Oversized official body was read before its header was rejected")
    def opener(request, **kwargs):
        calls.append(request.get_method())
        body = ForbiddenBody(BODY)
        body.status = 200
        body.headers = response(size=len(BODY) if method == "GET" and request.get_method() == "HEAD" else small.SMALL_LIMIT+1).headers
        return body
    monkeypatch.setattr(base.urllib.request, "urlopen", opener)
    request = base.urllib.request.Request(fixture.physio[0]["url"], method=method)
    bounded = small.SmallOfficialOpener(fixture.physio[0])
    if method == "GET":
        with bounded(base.urllib.request.Request(fixture.physio[0]["url"], method="HEAD"), timeout=5):
            pass
    with pytest.raises(base.BatchError):
        with bounded(request, timeout=5):
            pytest.fail("oversized body was accepted")
    assert calls == (["HEAD", "GET"] if method == "GET" else ["HEAD"])


def test_small_official_opener_preserves_exact_bounded_official_response(staging, monkeypatch):
    fixture = staging
    monkeypatch.setattr(base.urllib.request, "urlopen", lambda *a, **k: response(BODY))
    request = base.urllib.request.Request(fixture.physio[0]["url"], method="GET")
    bounded = small.SmallOfficialOpener(fixture.physio[0])
    with bounded(base.urllib.request.Request(fixture.physio[0]["url"], method="HEAD"), timeout=5):
        pass
    with bounded(request, timeout=5) as body:
        assert body.read() == BODY


@pytest.fixture
def software_scheduler(tmp_path, monkeypatch):
    """Explicit scheduler double, with real lease journals and native lock."""
    plan = base.build_plan()
    state = SimpleNamespace(lock=threading.Lock(), completed=set(), calls=[], heads=[], active=0, peak=0, large_peak=0,
        parser_active=0, parser_peak=0, nonphysio_finished=0, failures=set(), retained=set(),
        barrier=None, barrier_remaining=0, close_count=0, head_size=len(BODY))
    monkeypatch.setattr(base, "execution_gate", lambda root: CONTRACT)
    monkeypatch.setattr(base, "check_space", lambda *a: None)
    monkeypatch.setattr(parallel.shutil, "disk_usage", lambda *a: SimpleNamespace(free=1024**4))
    monkeypatch.setattr(parallel, "resource_preflight", lambda *a, **k: {"SOFTWARE_TEST_DOUBLE": True})
    monkeypatch.setattr(small, "resource_preflight", lambda *a, **k: {"SOFTWARE_TEST_DOUBLE": True})
    monkeypatch.setattr(base.urllib.request, "urlopen", lambda *a, **k: pytest.fail("scheduler fixture attempted real official I/O"))
    monkeypatch.setattr(base, "create_client", lambda: pytest.fail("scheduler ignored explicit software SDK"))
    base.atomic_json(tmp_path / "status/tasks.json", {"SOFTWARE_TEST_DOUBLE": True,
        "W1": {"status": "COMPLETE"}, "W2": {"status": "PENDING"}})
    resource_identity = base.digest({"schema": "source-raw-resource-identity-v2", "files": plan["files"],
        "adapter_code_sha256": plan["adapter_code_sha256"], "input_contract_sha256": CONTRACT})
    journal = resource(tmp_path / "audit/source_preparation/resource_progress.json", plan,
        identity_sha256=resource_identity)
    journal.close()
    def committed(entry):
        return {"SOFTWARE_TEST_DOUBLE": True, "status": "COMPLETE", "native_schema_audited": True,
            "identity": list(base.identity(entry)), "plan_sha256": plan["plan_sha256"],
            "input_contract_sha256": CONTRACT}
    def head(entry, **kwargs):
        with state.lock: state.heads.append(entry["object_id"])
        return {"http_status": 200, "bytes": state.head_size, "etag_is_sha256": False,
            "response_headers": response(size=state.head_size).headers}
    monkeypatch.setattr(base, "head_object", head)
    def verify(prior, root, *, entry, plan):
        with state.lock: return entry["object_id"] in state.completed
    monkeypatch.setattr(base, "verify_completed", verify)
    def native(dataset, path, **kwargs):
        with state.lock:
            state.parser_active += 1; state.parser_peak = max(state.parser_peak, state.parser_active)
        try:
            time.sleep(.001)
            return {"summary": {"SOFTWARE_TEST_DOUBLE": True}, "trials": []}
        finally:
            with state.lock: state.parser_active -= 1
    monkeypatch.setattr(base, "audit_native_source", native)
    def process(entry, *, root, batch_dir, plan, contract_sha256, budget, audit_fn, **kwargs):
        uid = entry["object_id"]
        with state.lock:
            if uid in state.completed:
                return committed(entry)
            state.calls.append(entry)
            if entry["dataset"] == "PhysionetMI":
                assert state.nonphysio_finished == 178, "Physio dispatched before all fixed non-Physio objects completed"
            state.active += 1; state.peak = max(state.peak, state.active)
            if entry["dataset"] != "PhysionetMI": state.large_peak = max(state.large_peak, state.active)
            wait_for_peers = entry["dataset"] == "PhysionetMI" and state.barrier_remaining > 0
            if wait_for_peers: state.barrier_remaining -= 1
        raw = base.raw_staging_path(root, plan, entry)
        raw.parent.mkdir(parents=True, exist_ok=True); raw.write_bytes(BODY)
        try:
            if wait_for_peers: state.barrier.wait(10)
            lease = budget.reserve_transfer(len(BODY), object_id=uid, probe_bytes=1)
            budget.consume(len(BODY), transfer_id=lease)
            budget.finish_transfer(lease, complete_body_read=True)
            audit_fn(entry["dataset"], raw)
            if uid in state.failures: raise RuntimeError("SOFTWARE_EXPECTED_TRANSFER_FAILURE")
            with state.lock:
                state.completed.add(uid)
                if entry["dataset"] != "PhysionetMI": state.nonphysio_finished += 1
            result = committed(entry)
            base.atomic_json(batch_dir / "objects" / uid / "state.json", result)
            return result
        finally:
            if uid not in state.retained: raw.unlink(missing_ok=True)
            with state.lock: state.active -= 1
    monkeypatch.setattr(base, "process_file", process)
    def aggregate(plan, *, root, batch_dir, **kwargs):
        assert state.active == 0, "aggregate ran before worker lifecycle drain"
        return {"status": "SOURCE_RAW_INVENTORY_COMPLETE" if len(state.completed) == 505 else "SOURCE_RAW_INVENTORY_PARTIAL",
            "actual_verified_objects": len(state.completed), "SOFTWARE_TEST_DOUBLE": True}
    monkeypatch.setattr(base, "aggregate_inventory", aggregate)
    client = SimpleNamespace(close=lambda: setattr(state, "close_count", state.close_count+1))
    def run(**kwargs):
        kwargs.setdefault("audit_fn", native)
        for entry in plan["files"]:
            if entry["object_id"] in state.completed:
                path = tmp_path / "audit/source_preparation/batches" / plan["plan_sha256"] / "objects" / entry["object_id"] / "state.json"
                if not path.exists(): base.atomic_json(path, committed(entry))
        return small.execute_plan(plan, root=tmp_path, client=client, bucket="SOFTWARE_NOT_REAL_BUCKET", **kwargs)
    return SimpleNamespace(root=tmp_path, plan=plan, state=state, run=run, client=client,
        folder=tmp_path / "audit/source_preparation/batches" / plan["plan_sha256"])


def test_all_178_nonphysio_are_serial_then_sixteen_small_bodies_one_parser(software_scheduler):
    fixture = software_scheduler
    fixture.state.barrier = threading.Barrier(16); fixture.state.barrier_remaining = 16
    before = json.dumps(fixture.plan, sort_keys=True)
    result = fixture.run(max_new_files=194)
    assert len(fixture.state.calls) == result["new_objects_attempted"] == 194
    assert all(entry["dataset"] != "PhysionetMI" for entry in fixture.state.calls[:178])
    assert all(entry["dataset"] == "PhysionetMI" for entry in fixture.state.calls[178:])
    assert fixture.state.large_peak == 1 and fixture.state.peak == 16 and fixture.state.parser_peak == 1 and fixture.state.active == 0
    assert result["actual_verified_objects"] == 194 and result["inventory_status"] == "SOURCE_RAW_INVENTORY_PARTIAL"
    assert result["download_bytes_read"] == result["download_bytes_charged"] == 194 * len(BODY)
    assert result["numerical_cache_created"] is False and result["training_started"] is False and result["w2_complete"] is False
    assert fixture.state.close_count == 0 and json.dumps(fixture.plan, sort_keys=True) == before
    amendment = json.loads((fixture.root / result["operational_amendment"]["path"]).read_text())
    assert amendment["nonphysio_phase"] == "serial_all_178_before_PhysionetMI"
    assert amendment["parallel_native_parsers"] == 1 and amendment["max_small_raw_bytes"] == small.SMALL_LIMIT
    assert json.loads((fixture.root / "status/tasks.json").read_text())["W2"]["status"] == "PENDING"


def test_batch_limit_does_not_skip_remaining_nonphysio_or_open_physio(software_scheduler):
    fixture = software_scheduler
    result = fixture.run(max_new_files=3)
    assert len(fixture.state.calls) == result["new_objects_attempted"] == 3
    assert fixture.state.peak == 1 and fixture.state.parser_peak == 1
    assert not any(row["dataset"] == "PhysionetMI" for row in fixture.state.calls)


def test_sixteen_existing_small_objects_leave_no_seventeenth_serial_staging_slot(software_scheduler):
    fixture = software_scheduler
    entries = [row for row in fixture.plan["files"] if row["dataset"] == "PhysionetMI"][:16]
    paths = [stage(fixture, entry, partial=True)[0] for entry in entries]
    result = fixture.run(max_new_files=1)
    assert not fixture.state.calls and not fixture.state.heads
    assert result["inventory_status"] == "SOURCE_RAW_INVENTORY_PARTIAL"
    assert result["download_bytes_read"] == result["download_bytes_charged"] == 0
    assert all(path.read_bytes() == BODY for path in paths)


def test_one_nonphysio_failure_does_not_unlock_physio_phase(software_scheduler):
    fixture = software_scheduler
    fixture.state.failures.add(next(row["object_id"] for row in fixture.plan["files"] if row["dataset"] != "PhysionetMI"))
    result = fixture.run(max_new_files=180)
    assert result["inventory_status"] == "SOURCE_RAW_INVENTORY_PARTIAL"
    assert not any(row["dataset"] == "PhysionetMI" for row in fixture.state.calls)
    assert fixture.state.active == 0


def test_Physio_future_failure_preserves_owned_partial_stops_third_and_drains_peer(software_scheduler, monkeypatch):
    """Two actual futures, durable leases and a controlled retained-body stop."""
    fixture = software_scheduler
    nonphysio = [row for row in fixture.plan["files"] if row["dataset"] != "PhysionetMI"]
    physio = [row for row in fixture.plan["files"] if row["dataset"] == "PhysionetMI"]
    fixture.state.completed.update(row["object_id"] for row in nonphysio)
    fixture.state.nonphysio_finished = 178
    original_process, original_write = base.process_file, base.atomic_json
    peer_inflight, release_peer, retained_checkpoint = threading.Event(), threading.Event(), threading.Event()
    dispatched, completed_peers = [], []
    partial = base.raw_staging_path(fixture.root, fixture.plan, physio[0])
    partial = partial.with_name(partial.name + ".partial")

    def progress_checkpoint(path, value):
        original_write(path, value)
        if (path.name == "batch_progress.json"
                and value.get("status") == "RETAINED_RAW_OR_PARTIAL_REQUIRES_RESUME_NO_STAGE_ADVANCE"):
            # The second future can finish only after the scheduler has
            # durably observed the retained partial and stopped dispatching.
            retained_checkpoint.set()
            release_peer.set()
    monkeypatch.setattr(base, "atomic_json", progress_checkpoint)

    def controlled(entry, *, root, batch_dir, plan, contract_sha256, budget, audit_fn, **kwargs):
        if entry["dataset"] != "PhysionetMI":
            return original_process(entry, root=root, batch_dir=batch_dir, plan=plan,
                contract_sha256=contract_sha256, budget=budget, audit_fn=audit_fn, **kwargs)
        with fixture.state.lock:
            dispatched.append(entry["object_id"])
            fixture.state.active += 1
        token = None
        raw = base.raw_staging_path(root, plan, entry)
        try:
            assert entry["object_id"] in {physio[0]["object_id"], physio[1]["object_id"]}, "third incomplete object was dispatched"
            token = budget.reserve_transfer(len(BODY), object_id=entry["object_id"], probe_bytes=1)
            if entry["object_id"] == physio[0]["object_id"]:
                assert peer_inflight.wait(10), "peer future was not already in flight"
                partial.parent.mkdir(parents=True, exist_ok=True)
                partial.write_bytes(BODY[:5])
                original_write(batch_dir / "objects" / entry["object_id"] / "state.json", {
                    "SOFTWARE_TEST_DOUBLE": True, "status": "DOWNLOADING", "temporary_raw_owned": True,
                    "identity": list(base.identity(entry)), "plan_sha256": plan["plan_sha256"],
                    "input_contract_sha256": contract_sha256, "raw_path": raw.relative_to(root).as_posix(),
                    "head_observation": {"http_status": 200, "bytes": len(BODY),
                        "response_headers": response(size=len(BODY)).headers}})
                budget.consume(5, transfer_id=token)
                budget.finish_transfer(token, complete_body_read=False); token = None
                raise RuntimeError("SOFTWARE_EXPECTED_Physio_BODY_FAILURE")
            peer_inflight.set()
            assert release_peer.wait(10), "scheduler failed to checkpoint retained-body stop before draining peer"
            raw.parent.mkdir(parents=True, exist_ok=True); raw.write_bytes(BODY)
            budget.consume(len(BODY), transfer_id=token)
            budget.finish_transfer(token, complete_body_read=True); token = None
            audit_fn(entry["dataset"], raw)
            state = {"SOFTWARE_TEST_DOUBLE": True, "status": "COMPLETE", "native_schema_audited": True,
                "identity": list(base.identity(entry)), "plan_sha256": plan["plan_sha256"],
                "input_contract_sha256": contract_sha256}
            original_write(batch_dir / "objects" / entry["object_id"] / "state.json", state)
            raw.unlink()
            with fixture.state.lock:
                fixture.state.completed.add(entry["object_id"]); completed_peers.append(entry["object_id"])
            return state
        finally:
            if token is not None: budget.finish_transfer(token, complete_body_read=False)
            with fixture.state.lock: fixture.state.active -= 1
    monkeypatch.setattr(base, "process_file", controlled)

    result = fixture.run(max_new_files=3, max_file_workers=2)
    assert retained_checkpoint.is_set() and peer_inflight.is_set()
    assert set(dispatched) == {physio[0]["object_id"], physio[1]["object_id"]} and len(dispatched) == 2
    assert completed_peers == [physio[1]["object_id"]] and fixture.state.active == 0
    assert partial.read_bytes() == BODY[:5]
    large, retained = small.inspect_staging(fixture.root, fixture.plan, fixture.folder,
        contract_sha256=CONTRACT, max_file_workers=2)
    assert not large and retained == [physio[0]]
    saved = json.loads((fixture.root / "audit/source_preparation/resource_progress.json").read_text())
    assert not any(a["status"] == parallel.OPEN for row in saved["byte_scopes"].values() for a in row["attempts"])
    assert saved["download_bytes_charged"] == (len(BODY)+1)+len(BODY)
    assert saved["download_bytes_read"] == 5+len(BODY)
    assert result["failed"] == 1 and result["inventory_status"] == "SOURCE_RAW_INVENTORY_PARTIAL"
    assert result["actual_verified_objects"] == 179 and result["w2_complete"] is False
    assert json.loads((fixture.root / "status/tasks.json").read_text())["W2"]["status"] == "PENDING"


def test_entry_headroom_pause_never_reaches_HEAD_SDK_or_worker(software_scheduler, monkeypatch):
    fixture = software_scheduler
    def paused(*args, **kwargs):
        raise parallel.ResourcePreflightPause("SOFTWARE_HEADROOM_PAUSE")
    monkeypatch.setattr(small, "resource_preflight", paused)
    monkeypatch.setattr(base, "create_client", lambda: pytest.fail("SDK before cgroup headroom gate"))
    with pytest.raises((base.BatchError, parallel.ResourcePreflightPause)):
        small.execute_plan(fixture.plan, root=fixture.root, max_new_files=1,
            bucket="SOFTWARE_NOT_REAL_BUCKET")
    assert not fixture.state.heads and not fixture.state.calls and fixture.state.active == 0


def test_running_admission_headroom_pause_drains_existing_workers(software_scheduler, monkeypatch):
    fixture = software_scheduler
    fixture.state.completed.update(row["object_id"] for row in fixture.plan["files"] if row["dataset"] != "PhysionetMI")
    fixture.state.nonphysio_finished = 178
    admissions = [0]
    def preflight(root, raw_bytes, *, max_file_workers):
        if raw_bytes == len(BODY):
            # The audit uses this same preflight too. Count main-thread
            # admission checks only, without modifying production APIs.
            if threading.current_thread() is threading.main_thread():
                admissions[0] += 1
                if admissions[0] == 4:
                    raise parallel.ResourcePreflightPause("SOFTWARE_ADMISSION_HEADROOM_PAUSE")
        return {"SOFTWARE_TEST_DOUBLE": True}
    monkeypatch.setattr(small, "resource_preflight", preflight)
    try:
        result = fixture.run(max_new_files=16)
        assert result["inventory_status"] == "SOURCE_RAW_INVENTORY_PARTIAL"
    except parallel.ResourcePreflightPause:
        pass  # An explicit pause may propagate after the executor has drained.
    assert admissions[0] == 4 and len(fixture.state.calls) <= 3 and fixture.state.active == 0
    saved = json.loads((fixture.root / "audit/source_preparation/resource_progress.json").read_text())
    assert not any(a["status"] == parallel.OPEN for row in saved["byte_scopes"].values() for a in row["attempts"])


def test_oversize_Physio_HEAD_cannot_reach_process_or_body_even_with_test_process_double(software_scheduler):
    fixture = software_scheduler
    fixture.state.completed.update(row["object_id"] for row in fixture.plan["files"] if row["dataset"] != "PhysionetMI")
    fixture.state.nonphysio_finished = 178
    fixture.state.head_size = small.SMALL_LIMIT+1
    result = fixture.run(max_new_files=1)
    assert fixture.state.heads and not fixture.state.calls
    assert result["inventory_status"] == "SOURCE_RAW_INVENTORY_PARTIAL"
    assert result["download_bytes_read"] == result["download_bytes_charged"] == 0


def test_completed_replay_keeps_original_namespace_and_never_HEADs_completed_objects(software_scheduler):
    fixture = software_scheduler
    nonphysio = [row for row in fixture.plan["files"] if row["dataset"] != "PhysionetMI"]
    physio = [row for row in fixture.plan["files"] if row["dataset"] == "PhysionetMI"]
    fixture.state.completed.update(row["object_id"] for row in nonphysio+[physio[0]])
    fixture.state.nonphysio_finished = 178
    result = fixture.run(max_new_files=1)
    assert [row["object_id"] for row in fixture.state.calls] == [physio[1]["object_id"]]
    assert physio[0]["object_id"] not in fixture.state.heads
    assert result["actual_verified_objects"] == 180 and result["plan_sha256"] == fixture.plan["plan_sha256"]


@pytest.mark.parametrize("kind", ["byte_only", "wall_only", "state_without_pair"])
def test_existing_accounting_cannot_be_recreated_from_zero(software_scheduler, monkeypatch, kind):
    fixture = software_scheduler
    resource_path = fixture.root / "audit/source_preparation/resource_progress.json"
    if kind == "byte_only":
        resource_path.with_name("resource_progress.wall.json").unlink()
        base.atomic_json(resource_path, {"SOFTWARE_TEST_DOUBLE": True})
    elif kind == "wall_only":
        resource_path.unlink()
        base.atomic_json(resource_path.with_name("resource_progress.wall.json"), {"SOFTWARE_TEST_DOUBLE": True})
    else:
        resource_path.unlink(); resource_path.with_name("resource_progress.wall.json").unlink()
        base.atomic_json(fixture.root / "audit/source_preparation/batches" / fixture.plan["plan_sha256"] /
            "objects/SOFTWARE_ORPHAN/state.json", {"SOFTWARE_TEST_DOUBLE": True})
    with pytest.raises(base.BatchError):
        fixture.run(max_new_files=1)
    assert not fixture.state.calls and not fixture.state.heads


def test_CLI_preserves_canonical_full_cache_and_R2_pipeline(tmp_path, monkeypatch):
    from tools import prepare_all_sources as pipeline
    calls = []
    def prepare(root, **kwargs):
        calls.append((root, kwargs))
        return {"status": "W2_COMPLETE_CACHE_AND_R2_VERIFIED", "SOFTWARE_TEST_DOUBLE": True}
    monkeypatch.setattr(pipeline, "prepare_all", prepare)
    assert small.main(["--root", str(tmp_path), "--execute", "--progress", "audit/fixture_progress.json"]) == 0
    root, options = calls[0]
    assert root == tmp_path and options["execute"] is True
    assert options["progress"] == "audit/fixture_progress.json"
    # CLI may bind its validated operational limits with a partial wrapper.
    raw_fn = options["raw_fn"]
    assert raw_fn is small.execute_plan or getattr(raw_fn, "func", None) is small.execute_plan
