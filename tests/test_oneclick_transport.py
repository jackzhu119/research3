import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tarfile
import types
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paper3_transport as transport


def sha(data):
    return hashlib.sha256(data).hexdigest()


def row(path, data, **extra):
    return dict(path=path, sha256=sha(data), bytes=len(data), **extra)


def fails(code):
    return pytest.raises(transport.TransportError, match="^" + code + "$")


@pytest.fixture(autouse=True)
def restore_runtime():
    # The launcher starts in a fresh stdlib-only process. Other test modules
    # may already have imported system dateutil/urllib3 before these wheel
    # fixtures run; isolate that process state, then restore it exactly.
    # Tests which explicitly inject a foreign dependency still exercise the
    # unchanged production fail-closed SDK origin check below.
    previous = transport._sdk_site
    paths = sys.path[:]
    bytecode = sys.dont_write_bytecode
    def sdk_module(name):
        return name.partition(".")[0] in transport.SDK_IMPORT_ROOTS
    modules = {name: module for name, module in sys.modules.items() if sdk_module(name)}
    for name in modules:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if sdk_module(name):
                sys.modules.pop(name, None)
        sys.modules.update(modules)
        transport._sdk_site = previous
        sys.path[:] = paths
        sys.dont_write_bytecode = bytecode


class FakeClient:
    def __init__(self, data=b"content", *, status=200, declared=None, exception=None):
        self.body = io.BytesIO(data)
        self.data = data
        self.status = status
        self.declared = len(data) if declared is None else declared
        self.exception = exception
        self.calls = []

    def get_object(self, **kwargs):
        self.calls.append(kwargs)
        if self.exception:
            raise self.exception
        return {"Body": self.body, "ContentLength": self.declared,
                "ResponseMetadata": {"HTTPStatusCode": self.status}}


def test_get_complete_commit_and_distinct_cache_receipt(tmp_path):
    data = b"actual-complete-object" * 50
    descriptor = row("archives/payload.tgz", data, key="paper3/fixed/payload.tgz")
    client = FakeClient(data)
    receipt = transport.fetch_object(client, "bucket-name", descriptor, tmp_path)
    target = tmp_path / descriptor["path"]
    assert target.read_bytes() == data
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert client.calls == [{"Bucket": "bucket-name", "Key": descriptor["key"]}]
    assert receipt["status"] == "R2_FULL_GET_SHA_VERIFIED"
    assert receipt["get_attempted"] and receipt["complete_body_read"]
    assert receipt["bytes_read"] == len(data)
    assert receipt["get_sha256"] == sha(data)
    assert client.body.closed
    cached_client = FakeClient(exception=RuntimeError("must not GET"))
    cached = transport.fetch_object(cached_client, "bucket-name", descriptor, tmp_path)
    assert cached["status"] == "LOCAL_VERIFIED_CACHE_HIT"
    assert cached["get_attempted"] is False and cached["bytes_read"] == 0
    assert cached["get_sha256"] is None and not cached_client.calls


@pytest.mark.parametrize("data,status,declared,code", [
    (b"content", 206, 7, "R2_GET_HTTP_NOT_200"),
    (b"content", 200, 8, "R2_GET_DECLARED_LENGTH_MISMATCH"),
    (b"short", 200, 7, "R2_GET_BODY_SHA_OR_LENGTH_MISMATCH"),
    (b"content-long", 200, 7, "R2_GET_BODY_LENGTH_MISMATCH"),
    (b"BAD-DAT", 200, 7, "R2_GET_BODY_SHA_OR_LENGTH_MISMATCH"),
])
def test_get_refuses_partial_or_wrong_bytes(tmp_path, data, status, declared, code):
    client = FakeClient(data, status=status, declared=declared)
    descriptor = row("data.bin", b"content", key="paper3/fixed/data.bin")
    with fails(code):
        transport.fetch_object(client, "bucket-name", descriptor, tmp_path)
    assert not (tmp_path / "data.bin").exists()
    assert not list(tmp_path.glob("*.partial"))
    assert client.body.closed


def test_get_existing_bad_local_is_never_replaced(tmp_path):
    (tmp_path / "data.bin").write_bytes(b"foreign")
    client = FakeClient()
    with fails("LOCAL_SHA_OR_BYTES_MISMATCH"):
        transport.fetch_object(client, "bucket-name",
                               row("data.bin", b"content", key="paper3/a"), tmp_path)
    assert not client.calls
    assert (tmp_path / "data.bin").read_bytes() == b"foreign"


def test_get_atomic_race_keeps_other_writer_file(tmp_path, monkeypatch):
    original = os.link

    def competing_writer(*args, **kwargs):
        (tmp_path / "data.bin").write_bytes(b"other writer")
        return original(*args, **kwargs)

    monkeypatch.setattr(transport.os, "link", competing_writer)
    with fails("ATOMIC_TARGET_ALREADY_EXISTS"):
        transport.fetch_object(FakeClient(), "bucket-name",
                               row("data.bin", b"content", key="paper3/a"), tmp_path)
    assert (tmp_path / "data.bin").read_bytes() == b"other writer"
    assert not list(tmp_path.glob("*.partial"))


def test_get_suppresses_service_exception(tmp_path):
    secret = "never expose dummy secret"
    with fails("R2_GET_FAILED") as failure:
        transport.fetch_object(FakeClient(exception=RuntimeError(secret)), "bucket-name",
                               row("a.bin", b"content", key="paper3/a"), tmp_path)
    assert secret not in str(failure.value)
    assert failure.value.__cause__ is None


class ProbeClient(FakeClient):
    def __init__(self, *, put_status=200, wrong_get=False):
        super().__init__()
        self.put_status = put_status
        self.wrong_get = wrong_get
        self.put_calls = []

    def put_object(self, **kwargs):
        self.put_calls.append(kwargs)
        self.data = kwargs["Body"]
        self.body = io.BytesIO(b"!" * len(self.data) if self.wrong_get else self.data)
        self.declared = len(self.data)
        return {"ResponseMetadata": {"HTTPStatusCode": self.put_status}}


def test_fresh_write_probe_roundtrip_is_explicitly_not_science(tmp_path):
    client = ProbeClient()
    closure = "a" * 64
    report = transport.probe_write(client, "bucket-name", tmp_path, {"closure_sha256": closure})
    sent = client.put_calls[0]
    assert sent["IfNoneMatch"] == "*"
    assert sent["Key"].startswith("paper3/oneclick-probes/" + closure[:16] + "/")
    assert json.loads(sent["Body"])["scientific_result"] is False
    assert len(sent["Body"]) == sent["ContentLength"]
    assert "bucket-name" not in sent["Body"].decode()
    assert report["status"] == "R2_WRITE_AND_FULL_GET_PROBE_VERIFIED"
    assert report["verification"]["get_attempted"] is True
    assert report["scientific_result"] is False and report["training_started"] is False
    assert report["probe_object_retained"] is True
    assert Path(report["local_verified_path"]).read_bytes() == sent["Body"]
    assert client.calls == [{"Bucket": "bucket-name", "Key": sent["Key"]}]


def test_probe_put_failure_never_claims_roundtrip_or_attempts_get(tmp_path):
    client = ProbeClient(put_status=403)
    with fails("R2_WRITE_PROBE_PUT_HTTP_NOT_200"):
        transport.probe_write(client, "bucket-name", tmp_path, {"closure_sha256": "a" * 64})
    assert not client.calls
    assert not list(tmp_path.rglob("*.json"))


def test_probe_get_sha_failure_never_claims_write_read_success(tmp_path):
    client = ProbeClient(wrong_get=True)
    with fails("R2_WRITE_PROBE_FULL_GET_FAILED"):
        transport.probe_write(client, "bucket-name", tmp_path, {"closure_sha256": "a" * 64})
    assert len(client.put_calls) == len(client.calls) == 1
    assert not list(tmp_path.rglob("*.json"))
    assert not list(tmp_path.rglob("*.partial"))


@pytest.mark.parametrize("location", ["root", "parent", "target"])
def test_get_rejects_symlink_locations(tmp_path, location):
    real = tmp_path / "real"
    real.mkdir()
    destination = tmp_path / "out"
    destination.mkdir()
    relative = "child/file.bin"
    if location == "root":
        destination.rmdir()
        destination.symlink_to(real, target_is_directory=True)
    elif location == "parent":
        (destination / "child").symlink_to(real, target_is_directory=True)
    else:
        (destination / "child").mkdir()
        (real / "file.bin").write_bytes(b"content")
        (destination / relative).symlink_to(real / "file.bin")
    client = FakeClient()
    with pytest.raises(transport.TransportError):
        transport.fetch_object(client, "bucket-name",
                               row(relative, b"content", key="paper3/a"), destination)
    assert not client.calls


TOP = "paper3_data_plan_603d299351fa6605"


def inputs(tmp_path, files):
    result = tmp_path / "inputs.json"
    rows = [row(name, data) for name, data in files.items()]
    result.write_text(json.dumps({
        "schema_version": "paper3-fixed-deployment-package-inputs-v1",
        "status": "LOCAL_EXACT_PAYLOAD_LIST_NOT_AN_ARCHIVE_OR_UPLOAD",
        "top_level_directory": TOP, "raw_EEG_embedded": False,
        "full_wheelhouse_embedded": False, "files": rows,
        "file_count": len(rows), "uncompressed_bytes": sum(len(x) for x in files.values()),
    }), encoding="utf-8")
    return result


def archive(tmp_path, entries, *, mutate=None, name="package.tgz"):
    result = tmp_path / name
    with tarfile.open(result, "w:gz", format=tarfile.PAX_FORMAT) as stream:
        for relative, data in entries:
            member = tarfile.TarInfo(TOP + "/" + relative)
            member.size = len(data)
            member.mode = 0o755 if Path(relative).suffix in {".py", ".sh"} else 0o644
            member.uid = member.gid = member.mtime = 0
            if mutate:
                mutate(member)
            stream.addfile(member, io.BytesIO(data) if member.isfile() else None)
    return result


def test_exact_archives_merge_overlap_and_verify_all_members(tmp_path):
    first = {"code/train.py": b"print('fixture only')\n", "reports/check.json": b"{}"}
    first_input = inputs(tmp_path, first)
    first_archive = archive(tmp_path, list(first.items()))
    destination = tmp_path / "out"
    report = transport.extract_exact(first_archive, first_input, destination, TOP)
    assert report["verified_members"] == report["new_atomic_copies"] == 2
    assert report["verified_payload_bytes"] == sum(map(len, first.values()))
    assert report["project_root"] == str(destination / TOP)
    assert report["individual_member_network_GET_claimed"] is False
    assert report["training_started"] is False and report["GPU_actions"] == 0
    assert stat.S_IMODE((destination / TOP / "code/train.py").stat().st_mode) == 0o755
    second = {"reports/check.json": b"{}", "reports/late.json": b'{"ok": true}'}
    second_input = inputs(tmp_path, second)
    second_archive = archive(tmp_path, list(second.items()), name="late.tgz")
    report = transport.extract_exact(second_archive, second_input, destination, TOP)
    assert report["new_atomic_copies"] == report["identical_existing_files"] == 1
    for relative, data in (first | second).items():
        assert (destination / TOP / relative).read_bytes() == data
    repeated = transport.extract_exact(second_archive, second_input, destination, TOP)
    assert repeated["new_atomic_copies"] == 0
    assert repeated["identical_existing_files"] == 2


@pytest.mark.parametrize("entries", [
    [("../escape", b"x")], [("/absolute", b"x")],
    [("a.txt", b"x"), ("extra.txt", b"extra")],
    [("a.txt", b"x"), ("a.txt", b"x")],
])
def test_archive_rejects_any_non_exact_member_set_before_writing(tmp_path, entries):
    descriptor = inputs(tmp_path, {"a.txt": b"x"})
    package = archive(tmp_path, entries)
    destination = tmp_path / "out"
    with fails("ARCHIVE_EXACT_MEMBER_SET_MISMATCH"):
        transport.extract_exact(package, descriptor, destination, TOP)
    assert not list(destination.rglob("*.txt"))


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE,
                                  tarfile.CHRTYPE, tarfile.DIRTYPE])
def test_archive_rejects_link_device_and_directory(tmp_path, kind):
    def change(member):
        member.type = kind
        member.linkname = "../../outside"
    descriptor = inputs(tmp_path, {"a.txt": b"x"})
    package = archive(tmp_path, [("a.txt", b"x")], mutate=change)
    with fails("ARCHIVE_FIXED_REGULAR_HEADER_REQUIRED"):
        transport.extract_exact(package, descriptor, tmp_path / "out", TOP)
    assert not (tmp_path / "out" / TOP / "a.txt").exists()


def test_archive_refuses_wrong_sha_and_existing_different_bytes(tmp_path):
    descriptor = inputs(tmp_path, {"a.txt": b"expected"})
    package = archive(tmp_path, [("a.txt", b"WRONGBYT")])
    destination = tmp_path / "out"
    with fails("ARCHIVE_MEMBER_SHA_OR_LENGTH_MISMATCH"):
        transport.extract_exact(package, descriptor, destination, TOP)
    assert not (destination / TOP / "a.txt").exists()
    assert not list(destination.rglob("*.partial"))
    package = archive(tmp_path, [("a.txt", b"expected")])
    (destination / TOP / "a.txt").write_bytes(b"other")
    with fails("LOCAL_SHA_OR_BYTES_MISMATCH"):
        transport.extract_exact(package, descriptor, destination, TOP)
    assert (destination / TOP / "a.txt").read_bytes() == b"other"


def test_archive_rejects_concatenated_hidden_gzip_payload(tmp_path):
    import gzip
    descriptor = inputs(tmp_path, {"a.txt": b"x"})
    package = archive(tmp_path, [("a.txt", b"x")])
    with package.open("ab") as stream:
        stream.write(gzip.compress(b"concealed nonzero payload"))
    with fails("ARCHIVE_NONZERO_TRAILING_DATA"):
        transport.extract_exact(package, descriptor, tmp_path / "out", TOP)
    assert not (tmp_path / "out" / TOP / "a.txt").exists()


@pytest.mark.parametrize("location", ["root", "parent", "target"])
def test_archive_rejects_unsafe_extraction_roots(tmp_path, location):
    descriptor = inputs(tmp_path, {"dir/a.txt": b"x"})
    package = archive(tmp_path, [("dir/a.txt", b"x")])
    destination = tmp_path / "out"
    real = tmp_path / "real"
    real.mkdir()
    if location == "root":
        destination.symlink_to(real, target_is_directory=True)
    else:
        (destination / TOP).mkdir(parents=True)
        if location == "parent":
            (destination / TOP / "dir").symlink_to(real, target_is_directory=True)
        else:
            (destination / TOP / "dir").mkdir()
            (real / "a.txt").write_bytes(b"x")
            (destination / TOP / "dir/a.txt").symlink_to(real / "a.txt")
    with pytest.raises(transport.TransportError):
        transport.extract_exact(package, descriptor, destination, TOP)


@pytest.mark.parametrize("fault,code", [
    ("counter", "EXACT_PACKAGE_COUNTERS_MISMATCH"),
    ("raw", "FROZEN_PACKAGE_SCOPE_REQUIRED"),
    ("duplicate", "DUPLICATE_EXACT_PACKAGE_PATH"),
    ("private", "PRIVATE_FILE_IN_PACKAGE_FORBIDDEN"),
    ("extra_key", "EXACT_FILE_DESCRIPTOR_REQUIRED"),
])
def test_package_inputs_reject_unfrozen_scope(tmp_path, fault, code):
    descriptor = inputs(tmp_path, {"a.txt": b"x"})
    value = json.loads(descriptor.read_text())
    if fault == "counter":
        value["file_count"] += 1
    elif fault == "raw":
        value["raw_EEG_embedded"] = True
    elif fault == "duplicate":
        value["files"].append(value["files"][0])
    elif fault == "private":
        value["files"][0]["path"] = "config/.env"
    else:
        value["files"][0]["extra"] = "unfrozen"
    descriptor.write_text(json.dumps(value))
    with fails(code):
        transport.extract_exact(tmp_path / "never-read.tgz", descriptor, tmp_path / "out", TOP)


def sdk_bundle(tmp_path, *, malicious=None):
    bundle = tmp_path / "bundle"
    (bundle / "sdk").mkdir(parents=True)
    rows = []
    for package in sorted(transport.SDK_NAMES):
        path = bundle / "sdk" / (package + "-1.0-py3-none-any.whl")
        with zipfile.ZipFile(path, "w") as stream:
            member = zipfile.ZipInfo(package + "/__init__.py")
            member.external_attr = (stat.S_IFREG | 0o644) << 16
            if malicious and package == "boto3":
                if malicious == "traversal":
                    member.filename = "../outside.py"
                else:
                    member.external_attr = (stat.S_IFLNK | 0o777) << 16
            stream.writestr(member, "# local fixture\n")
        rows.append(row("sdk/" + path.name, path.read_bytes()))
    return bundle, rows


def test_all_eight_pinned_sdk_wheels_install_without_pip(tmp_path):
    bundle, rows = sdk_bundle(tmp_path)
    work = tmp_path / "work"
    assert transport.ensure_sdk(bundle, rows, work) is None
    assert sys.path[0] == str(work / "sdk_site")
    assert transport._sdk_site == work / "sdk_site"
    for package in transport.SDK_NAMES:
        assert (work / "sdk_site" / package / "__init__.py").read_bytes() == b"# local fixture\n"
    transport.ensure_sdk(bundle, rows, work)
    (work / "sdk_site/boto3/__init__.py").write_text("tampered")
    with fails("LOCAL_SHA_OR_BYTES_MISMATCH"):
        transport.ensure_sdk(bundle, rows, work)


@pytest.mark.parametrize("malicious,code", [
    ("traversal", "UNSAFE_RELATIVE_PATH"),
    ("symlink", "SDK_LINK_OR_ENCRYPTED_MEMBER"),
])
def test_sdk_refuses_unsafe_zip_members(tmp_path, malicious, code):
    bundle, rows = sdk_bundle(tmp_path, malicious=malicious)
    with fails(code):
        transport.ensure_sdk(bundle, rows, tmp_path / "work")
    assert not (tmp_path / "outside.py").exists()


def test_sdk_pins_and_exact_eight_are_required(tmp_path):
    bundle, rows = sdk_bundle(tmp_path)
    with fails("EXACT_EIGHT_SDK_WHEELS_REQUIRED"):
        transport.ensure_sdk(bundle, rows[:-1], tmp_path / "work")
    (bundle / rows[0]["path"]).write_bytes(b"altered wheel")
    with fails("LOCAL_SHA_OR_BYTES_MISMATCH"):
        transport.ensure_sdk(bundle, rows, tmp_path / "work")


def test_client_passes_explicit_credentials_and_auto_region_without_network(tmp_path, monkeypatch):
    calls = []

    def config(**kwargs):
        return kwargs

    def client(service, **kwargs):
        calls.append((service, kwargs))
        return "synthetic-client"

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(
        client=client, __file__=str(tmp_path / "boto3/__init__.py")))
    monkeypatch.setitem(sys.modules, "botocore.config", types.SimpleNamespace(
        Config=config, __file__=str(tmp_path / "botocore/config.py")))
    transport._sdk_site = tmp_path
    assert transport.create_client("https://account.r2.cloudflarestorage.com/", "bucket-name",
                                   "dummy-access", "dummy-secret") == "synthetic-client"
    service, kwargs = calls[0]
    assert service == "s3" and kwargs["region_name"] == "auto"
    assert kwargs["aws_access_key_id"] == "dummy-access"
    assert kwargs["aws_secret_access_key"] == "dummy-secret"
    assert kwargs["endpoint_url"] == "https://account.r2.cloudflarestorage.com"
    assert "verify" not in kwargs and "proxies" not in kwargs["config"]


def test_preloaded_external_dependency_cannot_bypass_pinned_sdk(tmp_path, monkeypatch):
    bundle, rows = sdk_bundle(tmp_path)
    monkeypatch.setitem(sys.modules, "dateutil", types.SimpleNamespace(
        __file__=str(tmp_path / "foreign/dateutil/__init__.py")))
    with fails("LOADED_SDK_OUTSIDE_PINNED_SITE"):
        transport.ensure_sdk(bundle, rows, tmp_path / "work")


def test_client_cannot_fall_back_to_system_sdk(monkeypatch):
    transport._sdk_site = None
    with fails("PINNED_SDK_NOT_INITIALIZED"):
        transport.create_client("https://account.r2.cloudflarestorage.com", "bucket-name",
                                "dummy-access", "dummy-secret")


@pytest.mark.parametrize("endpoint", ["http://account.r2.cloudflarestorage.com",
    "https://account.r2.cloudflarestorage.com:notaport",
    "https://dummy-secret@account.r2.cloudflarestorage.com",
    "https://account.r2.cloudflarestorage.com/path", "https://unrelated.example"])
def test_client_rejects_endpoint_without_exposing_input(endpoint):
    with fails("R2_HTTPS_ENDPOINT_REQUIRED"):
        transport.create_client(endpoint, "bucket-name", "dummy-access", "dummy-secret")


def test_hash_refuses_symlinks(tmp_path):
    target = tmp_path / "file"
    target.write_bytes(b"x")
    link = tmp_path / "link"
    link.symlink_to(target)
    with fails("FILE_SYMLINK_OR_NON_REGULAR"):
        transport.hash_file(link)
    assert transport.hash_file(target) == (sha(b"x"), 1)
