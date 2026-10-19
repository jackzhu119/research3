"""SOFTWARE_TEST_DOUBLE only: tiny urllib bodies and fabricated evidence.

No test opens a real provider URL, credentials, SDK, EEG, worker, GPU or cache.
All replaced trust pins belong exclusively to each temporary fixture root.
"""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request
import urllib.error

import pytest

from tools import source_preparation_batch as base
from tests.test_source_preparation_batch import fake_backup, fake_native


BODY = b"SOFTWARE_TEST_DOUBLE_NOT_EDF_abcdef"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True, indent=2)+"\n")


def software_evidence(root):
    entries = [row for row in base.build_plan()["files"] if row["dataset"] == "PhysionetMI"]
    provider = root / "audit/source_schema/evidence/physionet_provider.html"
    sums = root / "audit/source_schema/evidence/physionet_SHA256SUMS.txt"
    evidence = root / "audit/source_schema/evidence_manifest.json"
    provider.parent.mkdir(parents=True)
    lines = ["SOFTWARE_TEST_DOUBLE provider documentation"] * 744
    lines.extend(["Download the files using AWS command line tools:",
        'aws s3 sync --no-sign-request s3://physionet-open/eegmmidb/1.0.0/ DESTINATION'])
    provider.write_text("\n".join(lines)+"\n")
    sums.write_text("".join(f"{sha(BODY)} S{int(row['provider_subject']):03d}/{row['file']}\n" for row in entries))
    descriptors = []
    for path, source in ((provider, "https://physionet.org/content/eegmmidb/1.0.0/"),
        (sums, "https://physionet.org/files/eegmmidb/1.0.0/SHA256SUMS.txt")):
        descriptors.append({"path": path.relative_to(root).as_posix(), "sha256": sha(path.read_bytes()),
            "bytes": path.stat().st_size, "source_url_or_derivation": source,
            "role": "source_metadata_or_fixed_adapter_evidence"})
    for index in range(20):
        extra = provider.parent / f"SOFTWARE_EVIDENCE_{index:02d}.txt"
        extra.write_bytes(f"SOFTWARE_TEST_DOUBLE_EVIDENCE_{index}".encode())
        descriptors.append({"path": extra.relative_to(root).as_posix(), "sha256": sha(extra.read_bytes()),
            "bytes": extra.stat().st_size, "source_url_or_derivation": "SOFTWARE_TEST_DOUBLE",
            "role": "source_metadata_or_fixed_adapter_evidence"})
    write_json(evidence, {"SOFTWARE_TEST_DOUBLE": True, "files": descriptors,
        "licences": {"PhysionetMI": {"value": "SOFTWARE_ONLY", "evidence": provider.name}}})
    return SimpleNamespace(root=root, entries=entries, entry=entries[0], provider=provider, sums=sums,
        evidence=evidence, descriptors=descriptors)


class SoftwareResponse(io.BytesIO):
    """Local response double; geturl deliberately has no network behavior."""
    def __init__(self, body=b"", *, status=200, size=None, final_url, content_range=None, extra_headers=None):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body) if size is None else size),
            "ETag": '"SOFTWARE_opaque_not_SHA256"', "Last-Modified": "Wed, 12 Oct 2022 07:29:37 GMT"}
        if content_range is not None: self.headers["Content-Range"] = content_range
        self.headers.update(extra_headers or {})
        self.final_url, self.read_calls = final_url, []

    def geturl(self):
        return self.final_url

    def getcode(self):
        return self.status

    def read(self, amount=-1):
        self.read_calls.append(amount)
        return super().read(amount)


@pytest.fixture
def official_fixture(tmp_path, monkeypatch):
    from tools import physionet_official_transport as official
    fixture = software_evidence(tmp_path)
    fixture.module = official
    def repin():
        manifest = json.loads(fixture.evidence.read_text())
        for row in manifest["files"]:
            path = fixture.root / row["path"]
            row.update(sha256=sha(path.read_bytes()), bytes=path.stat().st_size)
        write_json(fixture.evidence, manifest)
        monkeypatch.setattr(official, "EVIDENCE_MANIFEST_SHA256", sha(fixture.evidence.read_bytes()))
        monkeypatch.setattr(official, "PROVIDER_SHA256", sha(fixture.provider.read_bytes()))
        monkeypatch.setattr(official, "CHECKSUMS_SHA256", sha(fixture.sums.read_bytes()))
    fixture.repin = repin
    repin()
    fixture.catalog = official.load_official_catalog(fixture.root)
    return fixture


def test_catalog_is_exact_327_unsigned_official_mappings_and_full_retained_proof(official_fixture):
    fixture = official_fixture
    catalog = fixture.catalog
    assert len(catalog.entries) == 327
    expected = {f"S{int(row['provider_subject']):03d}/{row['file']}" for row in fixture.entries}
    assert set(catalog.entries) == expected
    for entry in fixture.entries:
        row = catalog.lookup(entry)
        assert row["canonical_url"] == entry["url"] and row["sha256"] == sha(BODY)
        assert row["relative_path"] in expected
        assert row["mirror_url"].startswith("https://physionet-open.s3.")
        assert row["mirror_url"].endswith("/eegmmidb/1.0.0/"+row["relative_path"])
    assert set(catalog.evidence) >= {"manifest", "provider", "checksums"}
    for item in catalog.evidence.values():
        path = fixture.root / item["path"]
        assert item["sha256"] == sha(path.read_bytes())


@pytest.mark.parametrize("target", ["manifest", "provider", "sums", "extra"])
def test_changed_retained_evidence_never_passes_fixed_complete_SHA_pins(official_fixture, target):
    fixture = official_fixture
    path = {"manifest": fixture.evidence, "provider": fixture.provider, "sums": fixture.sums,
        "extra": fixture.root / fixture.descriptors[-1]["path"]}[target]
    path.write_bytes(path.read_bytes()+b"TAMPER")
    with pytest.raises((ValueError, OSError)):
        fixture.module.load_official_catalog(fixture.root)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "wrong_digest", "unsafe_path"])
def test_catalog_rejects_semantic_checksum_corruption_even_with_fixture_pins_refreshed(official_fixture, kind):
    fixture = official_fixture
    lines = fixture.sums.read_text().splitlines()
    if kind == "missing": lines.pop()
    elif kind == "duplicate": lines.append(lines[0])
    elif kind == "wrong_digest": lines[0] = "not-a-SHA256 " + lines[0].split(" ", 1)[1]
    else: lines[0] = sha(BODY) + " ../S001/S001R04.edf"
    fixture.sums.write_text("\n".join(lines)+"\n"); fixture.repin()
    with pytest.raises(ValueError):
        fixture.module.load_official_catalog(fixture.root)


def test_provider_page_must_retain_exact_official_unsigned_S3_authority(official_fixture):
    fixture = official_fixture
    fixture.provider.write_text(fixture.provider.read_text().replace("--no-sign-request", "--SIGNED-NOT-APPROVED"))
    fixture.repin()
    with pytest.raises(ValueError):
        fixture.module.load_official_catalog(fixture.root)


@pytest.mark.parametrize("kind", ["other_run", "other_host", "query", "nonphysio"])
def test_catalog_lookup_does_not_expand_original_scientific_roster(official_fixture, kind):
    fixture = official_fixture
    entry = deepcopy(fixture.entry)
    if kind == "other_run": entry["file"] = "S001R03.edf"
    elif kind == "other_host": entry["url"] = entry["url"].replace("physionet.org", "external.invalid")
    elif kind == "query": entry["url"] += "?X-Amz-Credential=SOFTWARE_DENIED"
    else: entry = next(row for row in base.build_plan()["files"] if row["dataset"] != "PhysionetMI")
    with pytest.raises(ValueError): fixture.catalog.lookup(entry)


def software_network(fixture, monkeypatch, *, body=BODY, size=None, status=None, final_url=None, headers=None):
    requested, responses = [], []
    row = fixture.catalog.lookup(fixture.entry)
    def open_response(request, *, timeout):
        requested.append(request)
        method = request.get_method()
        offset = 0
        if request.get_header("Range"):
            offset = int(request.get_header("Range").split("=", 1)[1].split("-", 1)[0])
        payload = b"" if method == "HEAD" else body[offset:]
        observed_size = len(body) if method == "HEAD" else len(payload)
        response_status = 200 if method == "HEAD" else (206 if offset else 200)
        response = SoftwareResponse(payload, status=response_status if status is None else status,
            size=observed_size if size is None else size,
            final_url=final_url or row["mirror_url"],
            content_range=f"bytes {offset}-{len(body)-1}/{len(body)}" if offset else None,
            extra_headers=headers)
        responses.append(response)
        return response
    monkeypatch.setattr(fixture.module, "_unsigned_open", open_response)
    return requested, responses


def make_transport(fixture):
    return fixture.module.OfficialPhysionetTransport(fixture.root, fixture.entry, catalog=fixture.catalog)


@pytest.mark.parametrize("header", ["Authorization", "Cookie", "Proxy-Authorization", "X-Amz-Security-Token"])
@pytest.mark.parametrize("unredirected", [False, True])
def test_credential_headers_are_rejected_before_unsigned_network(official_fixture, monkeypatch, header, unredirected):
    fixture = official_fixture
    requested, _ = software_network(fixture, monkeypatch)
    request = urllib.request.Request(fixture.entry["url"], method="HEAD")
    add = request.add_unredirected_header if unredirected else request.add_header
    add(header, "SOFTWARE_FORBIDDEN_CREDENTIAL")
    with pytest.raises(ValueError): make_transport(fixture)(request, timeout=5)
    assert not requested


def test_HEAD_and_GET_preserve_canonical_identity_unsigned_URL_and_validators(official_fixture, monkeypatch):
    fixture = official_fixture
    requests, _ = software_network(fixture, monkeypatch)
    transport = make_transport(fixture)
    observation = base.head_object(fixture.entry, opener=transport)
    get = urllib.request.Request(fixture.entry["url"], headers={"If-Match": observation["response_headers"]["ETag"]})
    with transport(get, timeout=5) as response:
        assert response.read(7)+response.read() == BODY
        assert response.read(1) == b""
    assert [request.get_method() for request in requests] == ["HEAD", "GET"]
    assert all(request.full_url == fixture.catalog.lookup(fixture.entry)["mirror_url"] for request in requests)
    assert requests[1].get_header("If-match") == observation["response_headers"]["ETag"]
    for request in requests:
        assert not any(name.lower() in {"authorization", "cookie"} for name, _ in request.header_items())
    assert transport.receipt_path.is_file()
    serialized = json.dumps(transport.receipt())
    assert fixture.entry["url"] in serialized and fixture.catalog.lookup(fixture.entry)["mirror_url"] in serialized
    assert sha(fixture.sums.read_bytes()) in serialized


def test_unexpected_final_URL_is_rejected_before_body_read(official_fixture, monkeypatch):
    fixture = official_fixture
    _, responses = software_network(fixture, monkeypatch, final_url="https://external.invalid/not-approved.edf")
    with pytest.raises(ValueError):
        make_transport(fixture)(urllib.request.Request(fixture.entry["url"], method="HEAD"), timeout=5)
    assert not responses[0].read_calls


def test_GET_without_lifecycle_HEAD_is_rejected_before_network(official_fixture, monkeypatch):
    fixture = official_fixture
    requests, _ = software_network(fixture, monkeypatch)
    with pytest.raises(ValueError):
        make_transport(fixture)(urllib.request.Request(fixture.entry["url"]), timeout=5)
    assert not requests


@pytest.mark.parametrize("kind", ["status", "range", "length", "etag", "modified"])
def test_resume_frame_or_validator_mismatch_is_rejected_before_tail_read(official_fixture, monkeypatch, kind):
    fixture = official_fixture
    _, responses = software_network(fixture, monkeypatch)
    original_open = fixture.module._unsigned_open
    def changed(request, *, timeout):
        response = original_open(request, timeout=timeout)
        if request.get_method() == "GET":
            if kind == "status": response.status = 200
            elif kind == "range": response.headers["Content-Range"] = f"bytes 0-{len(BODY)-1}/{len(BODY)}"
            elif kind == "length": response.headers["Content-Length"] = str(len(BODY)-9)
            elif kind == "etag": response.headers["ETag"] = '"OTHER_OBJECT"'
            else: response.headers["Last-Modified"] = "OTHER_MODIFICATION"
        return response
    monkeypatch.setattr(fixture.module, "_unsigned_open", changed)
    transport = make_transport(fixture)
    observation = base.head_object(fixture.entry, opener=transport)
    request = urllib.request.Request(fixture.entry["url"], headers={"Range": "bytes=8-",
        "If-Match": observation["response_headers"]["ETag"]})
    with pytest.raises(ValueError): transport(request, timeout=5)
    assert len(responses) == 2 and not responses[-1].read_calls and responses[-1].closed


def test_truncated_EOF_cannot_claim_complete_GET(official_fixture, monkeypatch):
    fixture = official_fixture
    software_network(fixture, monkeypatch, body=BODY[:-1], size=len(BODY))
    transport = make_transport(fixture)
    observation = base.head_object(fixture.entry, opener=transport)
    request = urllib.request.Request(fixture.entry["url"],
        headers={"If-Match": observation["response_headers"]["ETag"]})
    with transport(request, timeout=5) as response:
        with pytest.raises(ValueError):
            while response.read(7):
                pass
    receipt = transport.receipt()
    assert receipt["events"][-1]["complete_body_read"] is False
    assert receipt["provider_complete_file_checksum"] is None


def test_bounded_overread_is_returned_for_original_meter_but_never_claims_complete_GET(official_fixture, monkeypatch):
    fixture = official_fixture
    software_network(fixture, monkeypatch, body=b"ABCD", size=3)
    transport = make_transport(fixture)
    observation = base.head_object(fixture.entry, opener=transport)
    request = urllib.request.Request(fixture.entry["url"],
        headers={"If-Match": observation["response_headers"]["ETag"]})
    with transport(request, timeout=5) as response:
        assert response.read(4) == b"ABCD"  # Original downloader must observe/meter all four bytes.
        with pytest.raises(ValueError): response.read(1)
    receipt = transport.receipt()
    assert receipt["events"][-1]["status"] == "GET_BODY_FRAME_FAILED"
    assert receipt["events"][-1]["bytes_read"] == 4 and receipt["events"][-1]["complete_body_read"] is False
    assert receipt["provider_complete_file_checksum"] is None


def test_original_downloader_meters_bounded_overread_once_and_preserves_failed_partial(official_fixture, monkeypatch):
    fixture = official_fixture
    software_network(fixture, monkeypatch, body=b"ABCD", size=3)
    transport = make_transport(fixture)
    observation = base.head_object(fixture.entry, opener=transport)
    plan = base.build_plan(max_download_bytes=100)
    final = base.raw_staging_path(fixture.root, plan, fixture.entry)
    partial = final.with_name(final.name + ".partial")
    partial.parent.mkdir(parents=True)
    budget = base.ResourceBudget(plan, fixture.root / "audit/SOFTWARE_overread_resource.json")
    try:
        with pytest.raises(base.BatchError, match="official_get_body_exceeds_observed_size"):
            base.download_object(fixture.entry, observation, partial, final, budget=budget, opener=transport)
        assert budget.data["download_bytes_read"] == 4 and budget.data["download_bytes_charged"] == 4
        ledger = budget.data["byte_scopes"][budget.scope]
        assert len(ledger["attempts"]) == 1 and ledger["active"] is None
        assert ledger["attempts"][0]["observed_bytes"] == 4
    finally:
        budget.close()
    assert not final.exists() and partial.is_file()
    # The original guard meters before writing an overrun block; retaining an
    # empty partial is truthful and never presents those bytes as a valid raw.
    assert partial.read_bytes() == b""
    event = transport.receipt()["events"][-1]
    assert event["bytes_read"] == 4 and event["status"] == "GET_BODY_FRAME_FAILED"
    assert event["complete_body_read"] is False


def test_early_close_or_zero_read_never_claims_complete_body_or_provider_SHA(official_fixture, monkeypatch):
    fixture = official_fixture
    software_network(fixture, monkeypatch)
    transport = make_transport(fixture)
    observation = base.head_object(fixture.entry, opener=transport)
    request = urllib.request.Request(fixture.entry["url"],
        headers={"If-Match": observation["response_headers"]["ETag"]})
    with transport(request, timeout=5) as response:
        assert response.read(0) == b""
        assert response.read(5) == BODY[:5]
    receipt = transport.receipt()
    assert receipt["events"][-1]["status"] == "GET_CLOSED_WITHOUT_COMPLETE_EOF"
    assert receipt["events"][-1]["complete_body_read"] is False
    assert receipt["provider_complete_file_checksum"] is None


@pytest.mark.parametrize("size", [0, 16*1024**2+1])
def test_HEAD_size_must_fit_fixed_small_object_cap_before_any_read(official_fixture, monkeypatch, size):
    fixture = official_fixture
    _, responses = software_network(fixture, monkeypatch, size=size)
    with pytest.raises(ValueError):
        base.head_object(fixture.entry, opener=make_transport(fixture))
    assert not responses[0].read_calls


def test_receipt_records_only_safe_headers_without_sensitive_response_values(official_fixture, monkeypatch):
    fixture = official_fixture
    software_network(fixture, monkeypatch, headers={"Set-Cookie": "SOFTWARE_SECRET_COOKIE",
        "Authorization": "SOFTWARE_SECRET_AUTH", "X-Amz-Security-Token": "SOFTWARE_SECRET_TOKEN"})
    transport = make_transport(fixture)
    base.head_object(fixture.entry, opener=transport)
    serialized = json.dumps(transport.receipt())
    assert "SOFTWARE_SECRET" not in serialized


@pytest.mark.parametrize("status", [302, 404, 429, 503])
def test_HTTPError_records_actual_URL_and_safe_headers_without_reading_private_body(official_fixture, monkeypatch, status):
    fixture = official_fixture
    mirror = fixture.catalog.lookup(fixture.entry)["mirror_url"]
    error_body = SoftwareResponse(b"SOFTWARE_PRIVATE_ERROR_BODY", final_url=mirror)
    safe_headers = {"Content-Length": "27", "Content-Type": "text/plain", "ETag": '"SOFTWARE_ERROR_ETAG"'}
    headers = {**safe_headers, "Set-Cookie": "SOFTWARE_PRIVATE_COOKIE", "Authorization": "SOFTWARE_PRIVATE_AUTH"}
    def failed(request, *, timeout):
        assert request.full_url == mirror
        raise urllib.error.HTTPError(request.full_url, status, "SOFTWARE_PRIVATE_REASON", headers, error_body)
    monkeypatch.setattr(fixture.module, "_unsigned_open", failed)
    transport = make_transport(fixture)
    with pytest.raises(ValueError):
        base.head_object(fixture.entry, opener=transport)
    event = transport.receipt()["events"][-1]
    assert event["canonical_url"] == fixture.entry["url"]
    assert event["requested_url"] == event["final_url"] == mirror
    assert event["http_status"] == status and event["response_headers"] == safe_headers
    assert event["redirect_response_not_followed"] is (300 <= status < 400)
    assert not error_body.read_calls and error_body.closed
    assert "SOFTWARE_PRIVATE" not in json.dumps(transport.receipt())


def test_public_catalog_entry_HEAD_and_evidence_bindings_cannot_be_replaced_or_mutated(official_fixture, monkeypatch):
    fixture = official_fixture
    software_network(fixture, monkeypatch)
    transport = make_transport(fixture)
    base.head_object(fixture.entry, opener=transport)
    for instance, name in ((transport, "entry"), (transport, "head"), (transport, "limit"),
        (fixture.catalog, "entries"), (fixture.catalog, "evidence")):
        with pytest.raises(AttributeError): setattr(instance, name, {})
    with pytest.raises(TypeError): transport.entry["mirror_url"] = "https://external.invalid/changed"
    with pytest.raises(TypeError): transport.head["bytes"] = 1
    relative = fixture.catalog.lookup(fixture.entry)["relative_path"]
    with pytest.raises(TypeError): fixture.catalog.entries[relative]["sha256"] = "f"*64
    with pytest.raises(TypeError): fixture.catalog.evidence["checksums"]["sha256"] = "f"*64
    detached = fixture.catalog.lookup(fixture.entry)
    detached["mirror_url"] = "https://external.invalid/changed"
    assert fixture.catalog.lookup(fixture.entry)["mirror_url"] == transport.entry["mirror_url"]


def test_original_downloader_resume_uses_actual_prefix_plus_tail_whole_SHA(official_fixture, monkeypatch):
    fixture = official_fixture
    requests, _ = software_network(fixture, monkeypatch)
    transport = make_transport(fixture)
    observation = base.head_object(fixture.entry, opener=transport)
    final = base.raw_staging_path(fixture.root, base.build_plan(), fixture.entry)
    partial = final.with_name(final.name + ".partial")
    partial.parent.mkdir(parents=True); partial.write_bytes(BODY[:8])
    result = base.download_object(fixture.entry, observation, partial, final, opener=transport)
    assert final.read_bytes() == BODY and result["sha256"] == sha(BODY)
    assert result["resumed_from_bytes"] == 8 and result["download_bytes_this_invocation"] == len(BODY)-8
    assert requests[-1].get_header("Range") == "bytes=8-"
    assert requests[-1].get_header("If-match") == observation["response_headers"]["ETag"]
    receipt = json.dumps(transport.receipt())
    assert "range_suffix_only_not_complete_file" in receipt
    assert sha(BODY[8:]) in receipt
    # The frozen downloader truthfully reports its own checksum scope; the
    # additional provider gate is exercised by the audit tests below.
    assert result["provider_sha256_verified"] is False


@pytest.mark.parametrize("bad_prefix", [False, True])
def test_full_original_process_checksum_gate_blocks_bad_resume_before_native_audit(official_fixture, monkeypatch, bad_prefix):
    fixture = official_fixture
    software_network(fixture, monkeypatch)
    transport = make_transport(fixture)
    observation = base.head_object(fixture.entry, opener=transport)
    plan = base.build_plan()
    batch = fixture.root / "audit/source_preparation/batches" / plan["plan_sha256"]
    final = base.raw_staging_path(fixture.root, plan, fixture.entry)
    partial = final.with_name(final.name + ".partial")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"Z"*8 if bad_prefix else BODY[:8])
    write_json(batch / "objects" / fixture.entry["object_id"] / "state.json", {
        "SOFTWARE_TEST_DOUBLE": True, "status": "FAILED", "identity": list(base.identity(fixture.entry)),
        "object_id": fixture.entry["object_id"], "plan_sha256": plan["plan_sha256"],
        "input_contract_sha256": "c"*64, "temporary_raw_owned": True,
        "raw_path": final.relative_to(fixture.root).as_posix(), "head_observation": observation})
    inner_calls = []
    def native(*args, **kwargs):
        inner_calls.append(kwargs["expected_sha256"])
        return fake_native(*args, **kwargs)
    checksum_audit = fixture.module.ProviderChecksumAudit(native, fixture.catalog,
        transport_lookup=lambda entry: transport)
    monkeypatch.setattr(base, "check_space", lambda *args: None)
    monkeypatch.setattr(base, "create_client", lambda: pytest.fail("software checksum test attempted an SDK"))
    state = base.process_file(fixture.entry, root=fixture.root, batch_dir=batch, plan=plan,
        contract_sha256="c"*64, opener=transport, audit_fn=checksum_audit, backup_fn=fake_backup)
    expected_body = b"Z"*8+BODY[8:] if bad_prefix else BODY
    assert state["raw_integrity"]["sha256"] == sha(expected_body)
    if bad_prefix:
        assert not inner_calls and state["status"] == "SCHEMA_BLOCKED"
        assert state["native_schema_audited"] is False and state["schema_blocked"] is True
        assert transport.receipt()["provider_complete_file_checksum"] is None
    else:
        assert inner_calls == [sha(BODY)] and state["status"] == "COMPLETE"
        assert state["native_schema_audited"] is True and state["schema_blocked"] is False
        audit = json.loads((fixture.root / state["adapter_audit"]["path"]).read_text())
        proof = audit["official_source_transport"]
        assert proof["receipt"]["sha256"] == sha((fixture.root / proof["receipt"]["path"]).read_bytes())
        assert audit["official_provider_checksum"]["actual_complete_file_sha256"] == sha(BODY)
    assert not partial.exists()
    assert state["backup_receipt"] and state["raw_integrity"]["provider_sha256_verified"] is False
    backup = json.loads((fixture.root / state["backup_manifest"]["path"]).read_text())
    assert len(backup["files"]) == 3  # The original raw + native JSON/JSONL transaction is retained.


def test_provider_audit_compares_whole_file_SHA_and_never_accepts_suffix_digest(official_fixture, tmp_path):
    fixture = official_fixture
    calls = []
    audit = fixture.module.ProviderChecksumAudit(lambda *a, **k: calls.append(k), fixture.catalog)
    path = tmp_path / fixture.entry["file"]
    for wrong in (sha(BODY[8:]), "f"*64):
        with pytest.raises(ValueError):
            audit("PhysionetMI", path, provider_subject=fixture.entry["provider_subject"],
                session_id=fixture.entry["session_id"], source_version=fixture.entry["version"], expected_sha256=wrong)
    assert not calls


@pytest.mark.parametrize("changed", [False, True])
def test_existing_COMPLETE_checksum_validation_is_read_only_and_full_file_only(official_fixture, changed):
    fixture = official_fixture
    state = {"SOFTWARE_TEST_DOUBLE": True, "status": "COMPLETE", "native_schema_audited": True,
        "identity": list(base.identity(fixture.entry)), "object_id": fixture.entry["object_id"],
        "raw_integrity": {"sha256": sha(BODY[8:]) if changed else sha(BODY),
            "bytes": len(BODY), "complete_body_read": True, "provider_sha256_verified": False}}
    before = deepcopy(state)
    if changed:
        with pytest.raises(ValueError):
            fixture.module.verify_completed_provider_checksum(fixture.entry, state, fixture.catalog)
    else:
        proof = fixture.module.verify_completed_provider_checksum(fixture.entry, state, fixture.catalog)
        assert sha(BODY) in json.dumps(proof)
    assert state == before


@pytest.mark.parametrize("bad_historical_digest", [False, True])
def test_aggregate_preflight_checks_every_existing_COMPLETE_without_raw_or_state_changes(official_fixture, bad_historical_digest):
    fixture = official_fixture
    plan = base.build_plan()
    batch = fixture.root / "audit/source_preparation/batches" / plan["plan_sha256"]
    paths = []
    for index, entry in enumerate(fixture.entries[:2]):
        path = batch / "objects" / entry["object_id"] / "state.json"
        write_json(path, {"SOFTWARE_TEST_DOUBLE": True, "status": "COMPLETE",
            "identity": list(base.identity(entry)), "plan_sha256": plan["plan_sha256"],
            "raw_integrity": {"complete_body_read": True, "sha256":
                sha(BODY[8:]) if bad_historical_digest and index == 1 else sha(BODY)}})
        paths.append(path)
    before = {path: path.read_bytes() for path in paths}
    if bad_historical_digest:
        with pytest.raises(ValueError):
            fixture.module.verify_all_completed_provider_checksums(fixture.root, plan, batch, fixture.catalog)
    else:
        result = fixture.module.verify_all_completed_provider_checksums(fixture.root, plan, batch, fixture.catalog)
        assert result["verified_complete_files"] == 2 and result["required_source_files"] == 327
        assert result["source_or_W2_completion_granted"] is False
    assert all(path.read_bytes() == before[path] for path in paths)
    assert not (fixture.root / "data").exists()


def test_checksum_wrapper_preserves_nonphysio_native_interface_without_extra_source_claims(official_fixture):
    fixture = official_fixture
    calls = []
    result = {"summary": {"SOFTWARE_TEST_DOUBLE": True}, "trials": []}
    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return result
    audit = fixture.module.ProviderChecksumAudit(native, fixture.catalog)
    observed = audit("BNCI2014_001", fixture.root / "SOFTWARE_NOT_EEG.mat", expected_sha256="d"*64)
    assert observed is result and len(calls) == 1
    assert "official_provider_checksum" not in observed["summary"]


@pytest.fixture
def integrated_scheduler(official_fixture, monkeypatch):
    """Actual downloader/ledger/three-object backup; only EEG/SDK are doubles.

    Historical COMPLETE fixtures stand in for the independently tested immutable
    receipt verifier. All 327 provider digest checks remain the real oracle.
    """
    from tools import source_preparation_small_files as small
    from tools import source_preparation_parallel as parallel
    fixture = official_fixture
    plan, contract = base.build_plan(), "d" * 64
    batch = fixture.root / "audit/source_preparation/batches" / plan["plan_sha256"]
    identity = base.digest({"schema": "source-raw-resource-identity-v2", "files": plan["files"],
        "adapter_code_sha256": plan["adapter_code_sha256"], "input_contract_sha256": contract})
    journal = base.ResourceBudget(plan, fixture.root / "audit/source_preparation/resource_progress.json",
        identity_sha256=identity)
    journal.close()
    real_verify, real_process = base.verify_completed, base.process_file
    historical, native_calls, aggregate_calls = set(), [], []

    def historical_state(entry):
        return {"SOFTWARE_TEST_DOUBLE": True, "status": "COMPLETE", "native_schema_audited": True,
            "object_id": entry["object_id"], "identity": list(base.identity(entry)),
            "plan_sha256": plan["plan_sha256"], "input_contract_sha256": contract,
            "raw_path": base.raw_staging_path(fixture.root, plan, entry).relative_to(fixture.root).as_posix(),
            "raw_integrity": {"bytes": len(BODY), "complete_body_read": True, "sha256": sha(BODY)}}

    for entry in plan["files"]:
        if entry["object_id"] != fixture.entry["object_id"]:
            historical.add(entry["object_id"])
            write_json(batch / "objects" / entry["object_id"] / "state.json", historical_state(entry))
    write_json(fixture.root / "status/tasks.json", {"SOFTWARE_TEST_DOUBLE": True,
        "W1": {"status": "COMPLETE"}, "W2": {"status": "PENDING"}})
    monkeypatch.setattr(base, "execution_gate", lambda root: contract)
    monkeypatch.setattr(base, "check_space", lambda *a: None)
    monkeypatch.setattr(small, "resource_preflight", lambda *a, **k: {"SOFTWARE_TEST_DOUBLE": True})
    monkeypatch.setattr(parallel.shutil, "disk_usage", lambda *a: SimpleNamespace(free=1024**4))
    monkeypatch.setattr(base, "create_client", lambda: pytest.fail("integration attempted a real SDK"))
    monkeypatch.setattr(base.urllib.request, "urlopen", lambda *a, **k: pytest.fail("canonical urllib escaped mirror mapping"))
    monkeypatch.setattr(base, "verify_completed", lambda state, root, *, entry, plan:
        True if entry["object_id"] in historical else real_verify(state, root, entry=entry, plan=plan))
    monkeypatch.setattr(base, "run_backup", lambda **kwargs: fake_backup(**{key: value for key, value in kwargs.items() if key != "bucket"}))

    def native(dataset, path, **kwargs):
        native_calls.append((dataset, path.name, kwargs))
        return fake_native(dataset, path, **kwargs)

    def aggregate(plan, *, root, batch_dir, **kwargs):
        aggregate_calls.append(True)
        return {"status": "SOFTWARE_TEST_DOUBLE_NOT_SCIENTIFIC_INVENTORY", "actual_verified_objects": 505}
    monkeypatch.setattr(base, "aggregate_inventory", aggregate)
    fixture.small, fixture.plan, fixture.batch = small, plan, batch
    fixture.historical, fixture.native_calls, fixture.aggregate_calls = historical, native_calls, aggregate_calls
    fixture.native, fixture.real_process, fixture.historical_state = native, real_process, historical_state
    fixture.run = lambda **kwargs: small.execute_plan(plan, root=fixture.root,
        client=SimpleNamespace(close=lambda: None), bucket="SOFTWARE_NOT_REAL_BUCKET", audit_fn=native, **kwargs)
    return fixture


def test_official_scheduler_uses_same_mirror_for_admission_internal_HEAD_GET_and_whole_SHA(integrated_scheduler, monkeypatch):
    fixture = integrated_scheduler
    requests, responses = software_network(fixture, monkeypatch)
    result = fixture.run(official_physionet_mirror=True, max_file_workers=2)
    methods = [request.get_method() for request in requests]
    assert methods == ["HEAD", "HEAD", "GET"]
    assert all(request.full_url == fixture.catalog.lookup(fixture.entry)["mirror_url"] for request in requests)
    assert len(fixture.native_calls) == 1 and fixture.native_calls[0][2]["expected_sha256"] == sha(BODY)
    state = json.loads((fixture.batch / "objects" / fixture.entry["object_id"] / "state.json").read_text())
    assert state["status"] == "COMPLETE" and state["raw_integrity"]["sha256"] == sha(BODY)
    audit = json.loads((fixture.root / state["adapter_audit"]["path"]).read_text())
    assert audit["official_provider_checksum"]["actual_complete_file_sha256"] == sha(BODY)
    assert audit["official_source_transport"]["proof"]["raw_body_transport"] == "official-s3-complete-file-GET-observed"
    assert len(json.loads((fixture.root / state["backup_manifest"]["path"]).read_text())["files"]) == 3
    assert result["official_provider_checksum_verification"]["verified_complete_files"] == 327
    assert result["w2_complete"] is False and result["training_started"] is False
    amendment = json.loads((fixture.root / result["operational_amendment"]["path"]).read_text())
    assert amendment["physionet_transport"] == "official-s3"
    assert amendment["native_reader_revision"]["revision_id"] == "physionet_original_tal_inward_v1"
    assert amendment["official_transport_helper"]["sha256"] == sha(Path(fixture.module.__file__).read_bytes())
    assert fixture.aggregate_calls == [True]


def test_official_all505_complete_with_one_wrong_historical_SHA_blocks_before_aggregate(integrated_scheduler, monkeypatch):
    fixture = integrated_scheduler
    fixture.historical.add(fixture.entry["object_id"])
    bad = fixture.historical_state(fixture.entry)
    bad["raw_integrity"]["sha256"] = sha(BODY[8:])
    path = fixture.batch / "objects" / fixture.entry["object_id"] / "state.json"
    write_json(path, bad)
    before = path.read_bytes()
    monkeypatch.setattr(fixture.module, "_unsigned_open", lambda *a, **k: pytest.fail("COMPLETE oracle opened network"))
    with pytest.raises(base.BatchError, match="completed_file_checksum_oracle_failed"):
        fixture.run(official_physionet_mirror=True)
    assert path.read_bytes() == before and not fixture.aggregate_calls and not fixture.native_calls
    assert json.loads((fixture.root / "status/tasks.json").read_text())["W2"]["status"] == "PENDING"


def test_official_wrong_downloaded_whole_SHA_is_quarantined_and_never_native_fit(integrated_scheduler, monkeypatch):
    fixture = integrated_scheduler
    software_network(fixture, monkeypatch, body=b"X" * len(BODY))
    result = fixture.run(official_physionet_mirror=True, max_file_workers=1)
    state = json.loads((fixture.batch / "objects" / fixture.entry["object_id"] / "state.json").read_text())
    assert state["status"] == "SCHEMA_BLOCKED" and not fixture.native_calls
    assert result["schema_blocked"] == 1
    assert result["official_provider_checksum_verification"]["verified_complete_files"] == 326
    assert result["w2_complete"] is False


def test_explicit_attempt_revision_delegates_full_original_process_kwargs(integrated_scheduler, monkeypatch):
    from tools import source_attempt_revision
    fixture = integrated_scheduler
    software_network(fixture, monkeypatch)
    registration = "audit/SOFTWARE_TEST_DOUBLE_attempt_revision.json"
    path = fixture.root / registration
    write_json(path, {"SOFTWARE_TEST_DOUBLE": True})
    seen = []

    class SoftwareProcessor:
        def __init__(self, registration_path, *, root, plan, contract_sha256):
            assert registration_path == registration and root == fixture.root and plan == fixture.plan
            self.descriptor = base.descriptor(root, path)
        def __call__(self, entry, **kwargs):
            seen.append((entry["object_id"], kwargs))
            return fixture.real_process(entry, **kwargs)
    monkeypatch.setattr(source_attempt_revision, "RevisionProcessor", SoftwareProcessor)
    result = fixture.run(official_physionet_mirror=True, attempt_revision=registration)
    assert len(seen) == 505
    active = next(kwargs for uid, kwargs in seen if uid == fixture.entry["object_id"])
    assert active["batch_dir"] == fixture.batch and active["plan"] == fixture.plan
    assert isinstance(active["opener"], fixture.small.SmallOfficialOpener)
    assert active["budget"].snapshot()["download_bytes_read"] == len(BODY)
    amendment = json.loads((fixture.root / result["operational_amendment"]["path"]).read_text())
    assert amendment["attempt_revision"] == base.descriptor(fixture.root, path)


def test_default_dispatcher_gets_explicit_revision_and_progress_starts_RUNNING(integrated_scheduler, monkeypatch):
    fixture = integrated_scheduler
    software_network(fixture, monkeypatch)
    revisions, statuses = [], []
    original_write = base.atomic_json

    def dispatch(dataset, path, *, revision=None, **kwargs):
        revisions.append(deepcopy(revision))
        return fake_native(dataset, path, **kwargs)

    def write(path, value):
        if path.name == "batch_progress.json": statuses.append(value["status"])
        return original_write(path, value)
    monkeypatch.setattr(fixture.small.source_dispatch, "audit_native_source", dispatch)
    monkeypatch.setattr(base, "atomic_json", write)
    result = fixture.small.execute_plan(fixture.plan, root=fixture.root,
        client=SimpleNamespace(close=lambda: None), bucket="SOFTWARE_NOT_REAL_BUCKET", official_physionet_mirror=True)
    amendment = json.loads((fixture.root / result["operational_amendment"]["path"]).read_text())
    assert revisions == [amendment["native_reader_revision"]]
    assert revisions[0] is not None and statuses[0] == "RUNNING"
    assert statuses[-1] == "ALL_CANDIDATES_ATTEMPTED"


def test_mirror_CLI_dryrun_only_binds_explicit_options_without_loading_catalog(tmp_path, monkeypatch):
    from tools import source_preparation_small_files as small
    from tools import prepare_all_sources as pipeline
    seen = []
    monkeypatch.setattr(small.official, "load_official_catalog", lambda *a: pytest.fail("dry-run read actual source evidence"))
    monkeypatch.setattr(pipeline, "prepare_all", lambda root, **kwargs:
        seen.append((root, kwargs)) or {"status": "DRY_RUN_SOFTWARE_TEST_DOUBLE"})
    assert small.main(["--root", str(tmp_path), "--official-physionet-mirror", "--attempt-revision", "audit/revision.json"]) == 0
    assert seen[0][1]["execute"] is False
    raw_fn = seen[0][1]["raw_fn"]
    assert raw_fn.keywords["official_physionet_mirror"] is True
    assert raw_fn.keywords["attempt_revision"] == "audit/revision.json"
