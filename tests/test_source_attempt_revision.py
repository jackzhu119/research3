"""SOFTWARE_TEST_DOUBLE only: no network, SDK, credentials, GPU or real EEG.

These tests exercise local retry bookkeeping and mocked byte streams. They are
not research downloads, R2 verification, schema evidence or training results.
"""
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import source_attempt_revision as revision
from tools import source_preparation_batch as base
from tests.test_source_preparation_batch import fake_backup, fake_native, response


CONTRACT = "a" * 64
BODY = b"SOFTWARE_TEST_DOUBLE_NOT_EEG_FOR_REVISION"
FRESH_ETAG = '"SOFTWARE_FRESH_MIRROR_VALIDATOR"'


def forbidden(*args, **kwargs):
    pytest.fail("Revision test attempted network, SDK, budget construction or a real native audit")


def sign(record):
    record.pop("revision_sha256", None)
    record["revision_sha256"] = base.digest(record)
    return record


@pytest.fixture(autouse=True)
def no_external_operations(monkeypatch):
    monkeypatch.setattr(base.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(base, "create_client", forbidden)
    monkeypatch.setattr(base, "run_backup", forbidden)
    monkeypatch.setattr(base, "audit_native_source", forbidden)
    monkeypatch.setattr(base, "ResourceBudget", type("NoBudgetConstruction", (), {
        "R2_BACKUP": base.ResourceBudget.R2_BACKUP,
        "__init__": forbidden,
    }))
    monkeypatch.setattr(base, "check_space", lambda *args: None)


@pytest.fixture
def fixture(tmp_path):
    plan = base.build_plan()
    physio = [e for e in plan["files"] if e["dataset"] == "PhysionetMI"]
    entry = physio[0]
    batch = tmp_path / "audit/source_preparation/batches" / plan["plan_sha256"]
    canonical = batch / "objects" / entry["object_id"] / "state.json"
    original = {
        "object_id": entry["object_id"], "identity": list(base.identity(entry)),
        "plan_sha256": plan["plan_sha256"], "input_contract_sha256": CONTRACT,
        "status": "FAILED", "error_code": "SOFTWARE_OLD_ATTEMPT_FAILURE",
        "head_observation": {"http_status": 200, "bytes": len(BODY),
            "response_headers": {"Content-Length": str(len(BODY)),
                "ETag": '"SOFTWARE_OLD_CANONICAL_PROVIDER"', "Last-Modified": "old"}},
    }
    base.atomic_json(canonical, original)
    preserved = batch / "preserved_attempts" / entry["object_id"] / "state.json"
    preserved.parent.mkdir(parents=True)
    preserved.write_bytes(canonical.read_bytes())
    partial = tmp_path / "data/source_preparation/quarantined" / (entry["object_id"] + ".partial")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"SOFTWARE_OLD_PARTIAL_MUST_NOT_RESUME")
    for name in ("adapter_audit.json", "trial_metadata.jsonl", "backup_manifest.json", "backup_receipt.json"):
        (canonical.parent / name).write_text("SOFTWARE_OLD_EVIDENCE_MUST_REMAIN_UNCHANGED\n")
    for name in ("resource_progress.json", "resource_progress.wall.json"):
        path = tmp_path / "audit/source_preparation" / name
        path.write_text("SOFTWARE_EXISTING_NONZERO_BUDGET_DO_NOT_REWRITE\n")
    implementations = []
    for relative in sorted(revision.IMPLEMENTATION_FILES):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((revision._RUNTIME_ROOT / relative).read_bytes())
        implementations.append(base.descriptor(tmp_path, path))
    record = sign({
        "schema_version": revision.SCHEMA, "plan_sha256": plan["plan_sha256"],
        "input_contract_sha256": CONTRACT, "fresh_provider_validators_required": True,
        "budgets_reset": False, "implementation_files": implementations,
        "objects": [{"object_id": entry["object_id"], "prior_state": base.descriptor(tmp_path, preserved),
            "quarantined_partial": base.descriptor(tmp_path, partial)}],
    })
    registration = batch / "retry_registration_SOFTWARE_ONLY.json"
    base.atomic_json(registration, record)
    return SimpleNamespace(root=tmp_path, plan=plan, entry=entry, physio=physio,
        batch=batch, canonical=canonical, preserved=preserved, partial=partial,
        registration=registration, record=record)


def processor(f):
    return revision.RevisionProcessor(f.registration.relative_to(f.root).as_posix(),
        root=f.root, plan=f.plan, contract_sha256=CONTRACT)


def call(f, worker, entry=None, **kwargs):
    options = dict(root=f.root, batch_dir=f.batch, plan=f.plan, contract_sha256=CONTRACT)
    options.update(kwargs)
    return worker(entry or f.entry, **options)


def opener(request, **kwargs):
    assert request.get_method() in {"HEAD", "GET"}
    return response(b"" if request.get_method() == "HEAD" else BODY,
        size=len(BODY), etag=FRESH_ETAG)


def run(f, worker=None, **kwargs):
    return call(f, worker or processor(f), opener=opener, audit_fn=fake_native, backup_fn=fake_backup, **kwargs)


def tree(path):
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def rewrite_registration(f, edit):
    edit(f.record)
    base.atomic_json(f.registration, sign(f.record))


def test_registration_is_read_only_and_original_plan_code_and_budget_unchanged(fixture):
    f = fixture
    before = tree(f.root)
    plan = deepcopy(f.plan)
    worker = processor(f)
    assert worker.retries.keys() == {f.entry["object_id"]}
    assert len(f.plan["files"]) == 505 and f.plan == plan
    assert tree(f.root) == before
    assert not worker.revision_batch.exists()


@pytest.mark.parametrize("field,value", [
    ("schema_version", "unknown"), ("plan_sha256", "f" * 64),
    ("input_contract_sha256", "f" * 64), ("fresh_provider_validators_required", False),
    ("budgets_reset", True),
])
def test_registration_identity_and_policy_cannot_be_re_signed_away(fixture, field, value):
    f = fixture
    rewrite_registration(f, lambda record: record.update({field: value}))
    with pytest.raises(base.BatchError, match="identity_or_policy"):
        processor(f)
    assert not (f.batch / "attempt_revisions").exists()


def test_registration_digest_is_required(fixture):
    f = fixture
    f.record["objects"][0]["extra_tamper"] = True
    base.atomic_json(f.registration, f.record)
    with pytest.raises(base.BatchError, match="identity_or_policy"):
        processor(f)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "other_file", "forged_actual_code"])
def test_exact_seven_actual_implementation_files_required(fixture, mode):
    f = fixture
    def edit(record):
        descriptors = record["implementation_files"]
        if mode == "missing":
            descriptors.pop()
        elif mode == "duplicate":
            descriptors[-1] = deepcopy(descriptors[0])
        elif mode == "other_file":
            descriptors[-1] = base.descriptor(f.root, f.preserved)
        else:
            path = f.root / descriptors[0]["path"]
            path.write_text("# SOFTWARE_TAMPERED_ACTUAL_CODE\n")
            descriptors[0] = base.descriptor(f.root, path)
    rewrite_registration(f, edit)
    with pytest.raises(base.BatchError, match="actual_implementation_binding"):
        processor(f)


@pytest.mark.parametrize("mode", ["none", "duplicate", "unknown", "nonphysio"])
def test_only_nonempty_unique_known_physionet_failed_registration(fixture, mode):
    f = fixture
    def edit(record):
        if mode == "none": record["objects"] = []
        elif mode == "duplicate": record["objects"] *= 2
        elif mode == "unknown": record["objects"][0]["object_id"] = "f" * 64
        else: record["objects"][0]["object_id"] = f.plan["files"][0]["object_id"]
    rewrite_registration(f, edit)
    with pytest.raises(base.BatchError):
        processor(f)


@pytest.mark.parametrize("status", ["COMPLETE", "BACKUP_VERIFIED", "PENDING", "RAW_VERIFIED"])
def test_completed_or_nonfailed_prior_state_is_never_registered(fixture, status):
    f = fixture
    value = json.loads(f.preserved.read_text())
    value["status"] = status
    base.atomic_json(f.preserved, value)
    f.canonical.write_bytes(f.preserved.read_bytes())
    rewrite_registration(f, lambda record: record["objects"][0].update(prior_state=base.descriptor(f.root, f.preserved)))
    before = tree(f.root)
    with pytest.raises(base.BatchError, match="only_exact_failed"):
        processor(f)
    assert tree(f.root) == before


@pytest.mark.parametrize("field,value", [
    ("identity", ["PhysionetMI", "wrong", "wrong", "wrong.edf"]),
    ("object_id", "f" * 64), ("plan_sha256", "f" * 64), ("input_contract_sha256", "f" * 64),
])
def test_preserved_prior_identity_cannot_be_re_signed(fixture, field, value):
    f = fixture
    prior = json.loads(f.preserved.read_text()); prior[field] = value
    base.atomic_json(f.preserved, prior); f.canonical.write_bytes(f.preserved.read_bytes())
    rewrite_registration(f, lambda record: record["objects"][0].update(prior_state=base.descriptor(f.root, f.preserved)))
    with pytest.raises(base.BatchError, match="real_result_identity"):
        processor(f)


def test_prior_descriptor_must_not_point_to_canonical_mutable_state(fixture):
    f = fixture
    rewrite_registration(f, lambda record: record["objects"][0].update(prior_state=base.descriptor(f.root, f.canonical)))
    with pytest.raises(base.BatchError, match="preserved_outside_active_state"):
        processor(f)


@pytest.mark.parametrize("target", ["registration", "preserved", "partial", "implementation", "canonical"])
def test_call_revalidates_evidence_including_init_to_call_changes(fixture, monkeypatch, target):
    f = fixture; worker = processor(f)
    monkeypatch.setattr(base, "process_file", forbidden)
    paths = {"registration": f.registration, "preserved": f.preserved,
        "partial": f.partial, "implementation": f.root / next(iter(revision.IMPLEMENTATION_FILES)),
        "canonical": f.canonical}
    paths[target].write_text('{"SOFTWARE_CHANGED_AFTER_CONSTRUCTION": true}\n')
    before = tree(f.root)
    with pytest.raises(base.BatchError):
        call(f, worker)
    assert tree(f.root) == before


@pytest.mark.parametrize("mode", ["unknown", "url", "version", "extra_field", "identity"])
def test_unknown_or_tampered_entry_never_reaches_original_worker(fixture, monkeypatch, mode):
    f = fixture; worker = processor(f); entry = deepcopy(f.entry)
    monkeypatch.setattr(base, "process_file", forbidden)
    if mode == "unknown": entry["object_id"] = "f" * 64
    elif mode == "url": entry["url"] += "?SOFTWARE_TAMPER"
    elif mode == "version": entry["version"] = "bad"
    elif mode == "extra_field": entry["injected"] = True
    else: entry["provider_subject"] = "wrong"
    with pytest.raises(base.BatchError, match="exact_plan_entry"):
        call(f, worker, entry)


@pytest.mark.parametrize("name", ["root", "batch_dir", "plan", "contract_sha256"])
def test_call_identity_arguments_are_exact(fixture, monkeypatch, name):
    f = fixture; worker = processor(f)
    monkeypatch.setattr(base, "process_file", forbidden)
    value = f.root.parent if name in {"root", "batch_dir"} else "wrong"
    with pytest.raises(base.BatchError, match="process_identity"):
        call(f, worker, **{name: value})


@pytest.mark.parametrize("artifact", ["raw", "partial"])
def test_old_active_staging_cannot_start_a_fresh_provider_revision(fixture, artifact):
    f = fixture; worker = processor(f)
    path = base.raw_staging_path(f.root, f.plan, f.entry)
    if artifact == "partial": path = path.with_name(path.name + ".partial")
    path.parent.mkdir(parents=True); path.write_bytes(b"SOFTWARE_OLD_RAW_OR_PARTIAL")
    before = tree(f.root)
    with pytest.raises(base.BatchError, match="preserved_elsewhere"):
        call(f, worker)
    assert tree(f.root) == before


def test_quarantined_partial_cannot_alias_active_partial(fixture):
    f = fixture
    active = base.raw_staging_path(f.root, f.plan, f.entry)
    active = active.with_name(active.name + ".partial")
    active.parent.mkdir(parents=True); active.write_bytes(f.partial.read_bytes())
    rewrite_registration(f, lambda record: record["objects"][0].update(quarantined_partial=base.descriptor(f.root, active)))
    with pytest.raises(base.BatchError, match="outside_active_staging"):
        processor(f)


@pytest.mark.parametrize("prior_status", ["FAILED", "SCHEMA_BLOCKED"])
def test_new_real_audit_receipt_paths_promoted_without_overwriting_old_evidence(fixture, prior_status):
    f = fixture
    if prior_status != "FAILED":
        value = json.loads(f.preserved.read_text()); value["status"] = prior_status
        base.atomic_json(f.preserved, value); f.canonical.write_bytes(f.preserved.read_bytes())
        rewrite_registration(f, lambda record: record["objects"][0].update(prior_state=base.descriptor(f.root, f.preserved)))
    worker = processor(f)
    immutable = {p: p.read_bytes() for p in [f.preserved, f.partial,
        *[p for p in f.canonical.parent.iterdir() if p.name != "state.json"],
        f.root / "audit/source_preparation/resource_progress.json",
        f.root / "audit/source_preparation/resource_progress.wall.json"]}
    original_plan = deepcopy(f.plan)
    result = run(f, worker)
    assert result["status"] == "COMPLETE" and result["head_observation"]["response_headers"]["ETag"] == FRESH_ETAG
    indexed = json.loads(f.canonical.read_text())
    state_path = worker.revision_batch / "objects" / f.entry["object_id"] / "state.json"
    assert indexed["attempt_revision_state"] == base.descriptor(f.root, state_path)
    assert indexed["attempt_revision"] == base.descriptor(f.root, f.registration)
    assert indexed["prior_attempt_state"] == base.descriptor(f.root, f.preserved)
    assert indexed["attempt_revision_staging"] == {}
    assert indexed["backup_receipt"]["path"].startswith(worker.revision_batch.relative_to(f.root).as_posix())
    assert indexed["adapter_audit"]["path"].startswith(worker.revision_batch.relative_to(f.root).as_posix())
    receipt = json.loads((f.root / indexed["backup_receipt"]["path"]).read_text())
    rows = receipt["objects"]
    raw = next(row for row in rows if row["role"] == "source_development_raw")
    assert raw["key"] == f"{base.PREFIX}{f.plan['plan_sha256']}/{f.entry['object_id']}/" + indexed["raw_path"]
    for row in rows:
        if row["role"] == "audit":
            old_path = f.canonical.parent.relative_to(f.root).as_posix() + "/" + Path(row["path"]).name
            assert row["key"] != receipt["prefix"] + old_path
            assert "/attempt_revisions/" in row["key"]
    assert {p: p.read_bytes() for p in immutable} == immutable
    assert f.plan == original_plan and len(f.plan["files"]) == 505
    assert base.verify_completed(indexed, f.root, entry=f.entry, plan=f.plan)
    before = tree(f.root)
    replay = call(f, processor(f), opener=forbidden, audit_fn=forbidden, backup_fn=forbidden)
    assert replay["status"] == "COMPLETE" and tree(f.root) == before


def interrupt_after_prefix(f, worker=None):
    offset = 7
    class Interrupted(io.BytesIO):
        def read(self, size=-1):
            if self.tell(): raise OSError("SOFTWARE_STREAM_INTERRUPTED")
            return super().read(min(size, offset))
    def interrupted(request, **kwargs):
        if request.get_method() == "HEAD": return response(size=len(BODY), etag=FRESH_ETAG)
        body = Interrupted(BODY)
        body.status, body.headers = 200, response(size=len(BODY), etag=FRESH_ETAG).headers
        return body
    with pytest.raises(base.BatchError, match="operation_failed_details_suppressed"):
        call(f, worker or processor(f), opener=interrupted, audit_fn=forbidden, backup_fn=forbidden)
    raw = base.raw_staging_path(f.root, f.plan, f.entry)
    partial = raw.with_name(raw.name + ".partial")
    assert partial.read_bytes() == BODY[:offset]
    indexed = json.loads(f.canonical.read_text())
    assert indexed["status"] == "FAILED" and indexed["attempt_revision_staging"] == {"partial": base.descriptor(f.root, partial)}
    assert indexed["head_observation"]["response_headers"]["ETag"] == FRESH_ETAG
    return offset, partial


def test_failed_real_state_is_promoted_and_exact_revision_range_recovers(fixture):
    f = fixture; worker = processor(f)
    old = f.preserved.read_bytes(), f.partial.read_bytes()
    offset, partial = interrupt_after_prefix(f, worker)
    requests = []
    def resume(request, **kwargs):
        requests.append(request)
        if request.get_method() == "HEAD": return response(size=len(BODY), etag=FRESH_ETAG)
        assert request.get_header("Range") == f"bytes={offset}-"
        assert request.get_header("If-match") == FRESH_ETAG
        return response(BODY[offset:], status=206, content_range=f"bytes {offset}-{len(BODY)-1}/{len(BODY)}", etag=FRESH_ETAG)
    state = call(f, processor(f), opener=resume, audit_fn=fake_native, backup_fn=fake_backup)
    assert state["status"] == "COMPLETE" and state["raw_integrity"]["resumed_from_bytes"] == offset
    assert [r.get_method() for r in requests] == ["HEAD", "GET"] and not partial.exists()
    assert (f.preserved.read_bytes(), f.partial.read_bytes()) == old


@pytest.mark.parametrize("mutation", ["changed_partial", "changed_head", "bad_range"])
def test_changed_partial_provider_or_inexact_range_is_preserved_and_refused(fixture, mutation):
    f = fixture; _, partial = interrupt_after_prefix(f)
    saved = partial.read_bytes()
    if mutation == "changed_partial": partial.write_bytes(b"TAMPER!")
    before = f.canonical.read_bytes()
    def bad(request, **kwargs):
        if mutation == "changed_partial": pytest.fail("tampered partial reached provider")
        if request.get_method() == "HEAD":
            return response(size=len(BODY), etag='"SOFTWARE_CHANGED"' if mutation == "changed_head" else FRESH_ETAG)
        return response(BODY, etag=FRESH_ETAG)  # A 200 cannot resume an existing partial.
    with pytest.raises(base.BatchError):
        call(f, processor(f), opener=bad, audit_fn=forbidden, backup_fn=forbidden)
    assert partial.read_bytes() == (b"TAMPER!" if mutation == "changed_partial" else saved)
    if mutation == "changed_partial": assert f.canonical.read_bytes() == before
    else: assert json.loads(f.canonical.read_text())["status"] == "FAILED"


@pytest.mark.parametrize("mode", ["canonical_extra", "canonical_status", "revision_state"])
def test_indexed_canonical_and_actual_revision_state_must_match_exactly(fixture, mode):
    f = fixture; run(f)
    indexed = json.loads(f.canonical.read_text())
    if mode == "revision_state":
        (f.root / indexed["attempt_revision_state"]["path"]).write_text('{"SOFTWARE_TAMPER": true}\n')
    else:
        indexed["injected" if mode == "canonical_extra" else "status"] = True if mode == "canonical_extra" else "FAILED"
        base.atomic_json(f.canonical, indexed)
    before = tree(f.root)
    with pytest.raises(base.BatchError): processor(f)
    assert tree(f.root) == before


def test_unindexed_preexisting_revision_state_is_not_adopted(fixture):
    f = fixture; worker = processor(f)
    path = worker.revision_batch / "objects" / f.entry["object_id"] / "state.json"
    base.atomic_json(path, json.loads(f.preserved.read_text()))
    with pytest.raises(base.BatchError, match="unindexed_state"):
        call(f, worker)


def test_concurrent_canonical_mutation_is_preserved_after_real_worker_result(fixture):
    f = fixture; worker = processor(f)
    concurrent = b'{"SOFTWARE_CONCURRENT_CANONICAL_MUTATION_MUST_REMAIN":true}\n'
    def backup(**kwargs):
        result = fake_backup(**kwargs)
        f.canonical.write_bytes(concurrent)
        return result
    with pytest.raises(base.BatchError, match="concurrent_canonical_mutation_preserved"):
        call(f, worker, opener=opener, audit_fn=fake_native, backup_fn=backup)
    assert f.canonical.read_bytes() == concurrent
    revised = worker.revision_batch / "objects" / f.entry["object_id"] / "state.json"
    assert json.loads(revised.read_text())["status"] == "COMPLETE"
    assert (revised.parent / "backup_receipt.json").is_file()


def test_worker_cannot_promote_a_different_real_result_identity(fixture, monkeypatch):
    f = fixture; worker = processor(f); before = f.canonical.read_bytes()
    def wrong(entry, **kwargs):
        state = json.loads(f.preserved.read_text()); state["object_id"] = "f" * 64
        base.atomic_json(kwargs["batch_dir"] / "objects" / entry["object_id"] / "state.json", state)
        return state
    monkeypatch.setattr(base, "process_file", wrong)
    with pytest.raises(base.BatchError, match="real_result_identity"):
        call(f, worker)
    assert f.canonical.read_bytes() == before


def test_implementation_changed_during_real_worker_call_blocks_promotion(fixture, monkeypatch):
    f = fixture; worker = processor(f); before = f.canonical.read_bytes()
    def changed(entry, **kwargs):
        state = json.loads(f.preserved.read_text())
        base.atomic_json(kwargs["batch_dir"] / "objects" / entry["object_id"] / "state.json", state)
        (f.root / "tools/source_attempt_revision.py").write_text("# SOFTWARE_CHANGED_DURING_CALL\n")
        return state
    monkeypatch.setattr(base, "process_file", changed)
    with pytest.raises(base.BatchError, match="evidence_integrity"):
        call(f, worker)
    assert f.canonical.read_bytes() == before


def test_unregistered_exact_candidates_use_original_worker_and_same_budget_object(fixture, monkeypatch):
    f = fixture; worker = processor(f); budget = object(); calls = []
    def original(entry, **kwargs):
        calls.append((entry, kwargs)); return {"SOFTWARE_ORIGINAL_WORKER_RESULT": True}
    monkeypatch.setattr(base, "process_file", original)
    for entry in [f.plan["files"][0], f.physio[1]]:
        assert call(f, worker, entry, budget=budget)["SOFTWARE_ORIGINAL_WORKER_RESULT"]
    assert all(kwargs["batch_dir"] == f.batch and kwargs["budget"] is budget for _, kwargs in calls)
    assert not worker.revision_batch.exists()


def test_unregistered_failed_mirror_attempt_keeps_original_range_recovery(fixture):
    f = fixture; entry = f.physio[1]; offset = 5
    canonical = f.batch / "objects" / entry["object_id"] / "state.json"
    raw = base.raw_staging_path(f.root, f.plan, entry)
    partial = raw.with_name(raw.name + ".partial")
    partial.parent.mkdir(parents=True); partial.write_bytes(BODY[:offset])
    observation = base.head_object(entry, opener=opener)
    base.atomic_json(canonical, {"status": "FAILED", "object_id": entry["object_id"],
        "identity": list(base.identity(entry)), "plan_sha256": f.plan["plan_sha256"],
        "input_contract_sha256": CONTRACT, "head_observation": observation,
        "raw_path": raw.relative_to(f.root).as_posix(), "temporary_raw_owned": True})
    def resume(request, **kwargs):
        if request.get_method() == "HEAD": return response(size=len(BODY), etag=FRESH_ETAG)
        assert request.get_header("Range") == f"bytes={offset}-"
        return response(BODY[offset:], status=206,
            content_range=f"bytes {offset}-{len(BODY)-1}/{len(BODY)}", etag=FRESH_ETAG)
    state = call(f, processor(f), entry, opener=resume, audit_fn=fake_native, backup_fn=fake_backup)
    indexed = json.loads(canonical.read_text())
    assert state["status"] == "COMPLETE" and indexed["raw_integrity"]["resumed_from_bytes"] == offset
    assert "attempt_revision" not in indexed and "attempt_revisions" not in indexed["backup_receipt"]["path"]


def cache_restore_item(f, state):
    return dict(f.entry, object_uid=f.entry["object_id"], path=state["raw_path"],
        sha256=state["raw_integrity"]["sha256"], bytes=state["raw_integrity"]["bytes"],
        backup_receipt=state["backup_receipt"], temporary_raw_owned=True,
        raw_storage="verified_r2_only")


def test_completed_cache_kill_recovery_updates_real_revision_and_canonical_index(fixture, monkeypatch):
    from tools.prepare_source_cache import begin_owned_cache_restore
    f = fixture
    run(f)
    indexed = json.loads(f.canonical.read_text())
    revision_path = f.root / indexed["attempt_revision_state"]["path"]
    previous_revision_sha = indexed["attempt_revision_state"]["sha256"]
    immutable = {p: p.read_bytes() for p in revision_path.parent.iterdir() if p.name != "state.json"}
    marker = begin_owned_cache_restore(f.root, cache_restore_item(f, indexed))
    raw = f.root / indexed["raw_path"]
    raw.write_bytes(BODY)  # A mock cache GET committed just before SIGKILL.
    canonical_before = f.canonical.read_bytes()
    before_init = tree(f.root)
    worker = processor(f)
    assert tree(f.root) == before_init  # Ownership admission remains read-only.
    actual_recover = base.recover_interrupted_cache_restore
    calls = []
    def recover(state, **kwargs):
        assert kwargs["state_path"] == revision_path
        value = actual_recover(state, **kwargs)
        calls.append(value)
        # The real base recovery only writes the revision state. Promotion below
        # remains responsible for updating the independently checked index.
        assert f.canonical.read_bytes() == canonical_before
        return value
    monkeypatch.setattr(base, "recover_interrupted_cache_restore", recover)
    result = call(f, worker, opener=forbidden, audit_fn=forbidden, backup_fn=forbidden)
    assert calls == [True] and result["status"] == "COMPLETE"
    assert result["recovered_owned_cache_restore"] is True and not raw.exists()
    assert json.loads(marker.read_text())["status"] == "RAW_REMOVED_AFTER_VERIFIED_BACKUP"
    revised = json.loads(revision_path.read_text())
    indexed = json.loads(f.canonical.read_text())
    assert revised["recovered_owned_cache_restore"] is True
    assert indexed["recovered_owned_cache_restore_utc"] == revised["recovered_owned_cache_restore_utc"]
    assert indexed["attempt_revision_state"] == base.descriptor(f.root, revision_path)
    assert indexed["attempt_revision_state"]["sha256"] != previous_revision_sha
    assert indexed["attempt_revision_staging"] == {}
    assert {p: p.read_bytes() for p in immutable} == immutable
    before_replay = tree(f.root)
    # A new processor accepts the updated provenance and a completed replay
    # invokes no provider, backup or native-audit operation.
    monkeypatch.setattr(base, "recover_interrupted_cache_restore", actual_recover)
    assert call(f, processor(f), opener=forbidden, audit_fn=forbidden, backup_fn=forbidden)["status"] == "COMPLETE"
    assert tree(f.root) == before_replay


@pytest.mark.parametrize("condition", ["missing_marker", "changed_body", "removed_marker", "extra_partial"])
def test_reappeared_completed_raw_requires_exact_active_cache_ownership(fixture, condition):
    from tools.prepare_source_cache import begin_owned_cache_restore, recover_owned_cache_restore
    f = fixture
    run(f)
    state = json.loads(f.canonical.read_text())
    item = cache_restore_item(f, state)
    raw = f.root / state["raw_path"]
    if condition != "missing_marker": begin_owned_cache_restore(f.root, item)
    raw.write_bytes(BODY)
    if condition == "changed_body": raw.write_bytes(b"SOFTWARE_CHANGED_RESTORED_RAW")
    elif condition == "removed_marker":
        assert recover_owned_cache_restore(f.root, item) is True
        raw.write_bytes(BODY)
    elif condition == "extra_partial":
        raw.with_name(raw.name + ".partial").write_bytes(b"SOFTWARE_UNEXPECTED_PARTIAL")
    before = tree(f.root)
    with pytest.raises(base.BatchError):
        call(f, processor(f), opener=forbidden, audit_fn=forbidden, backup_fn=forbidden)
    assert tree(f.root) == before


def test_completed_cache_recovery_still_preserves_concurrent_canonical_mutation(fixture, monkeypatch):
    from tools.prepare_source_cache import begin_owned_cache_restore
    f = fixture
    run(f)
    state = json.loads(f.canonical.read_text())
    begin_owned_cache_restore(f.root, cache_restore_item(f, state))
    raw = f.root / state["raw_path"]
    raw.write_bytes(BODY)
    worker = processor(f)
    actual_recover = base.recover_interrupted_cache_restore
    concurrent = b'{"SOFTWARE_CONCURRENT_DURING_COMPLETE_RECOVERY":true}\n'
    def recover(state, **kwargs):
        result = actual_recover(state, **kwargs)
        f.canonical.write_bytes(concurrent)
        return result
    monkeypatch.setattr(base, "recover_interrupted_cache_restore", recover)
    with pytest.raises(base.BatchError, match="concurrent_canonical_mutation_preserved"):
        call(f, worker, opener=forbidden, audit_fn=forbidden, backup_fn=forbidden)
    assert f.canonical.read_bytes() == concurrent
    revised = f.root / state["attempt_revision_state"]["path"]
    assert json.loads(revised.read_text())["recovered_owned_cache_restore"] is True
