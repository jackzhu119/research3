"""Pinned local SDK, complete R2 GET and exact-list archive extraction.

This module never prints service exceptions or credentials and never trains.
The caller supplies the already frozen object and package descriptors.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import tempfile
import uuid
import zipfile

CHUNK = 1024 * 1024
SDK_NAMES = frozenset({"boto3", "botocore", "s3transfer", "jmespath", "urllib3",
                       "python_dateutil", "six", "certifi"})
SDK_IMPORT_ROOTS = SDK_NAMES - {"python_dateutil"} | {"dateutil"}
_sdk_site: Path | None = None


class TransportError(ValueError):
    """Only a fixed, nonsensitive code is exposed to the caller."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _need(condition, code):
    if not condition:
        raise TransportError(code)


def _relative(value):
    _need(isinstance(value, str) and value and "\\" not in value
          and not any(ord(c) < 32 for c in value), "UNSAFE_RELATIVE_PATH")
    path = PurePosixPath(value)
    _need(not path.is_absolute() and path.as_posix() == value
          and all(part not in {"", ".", ".."} for part in value.split("/")),
          "UNSAFE_RELATIVE_PATH")
    return path


def _root(value, *, create=False):
    path = Path(os.path.abspath(os.fspath(value)))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            _need(create, "DIRECTORY_MISSING")
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                pass
            mode = current.lstat().st_mode
        _need(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode),
              "DIRECTORY_SYMLINK_OR_NON_DIRECTORY")
    return path


def _file(root, relative, *, create_parents=False):
    root = _root(root)
    parts = _relative(relative).parts
    parent = root.joinpath(*parts[:-1])
    _root(parent, create=create_parents)
    path = parent / parts[-1]
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return path
    _need(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), "FILE_SYMLINK_OR_NON_REGULAR")
    return path


def _identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def hash_file(path):
    """Return (whole SHA256, bytes), refusing symlinks and changing files."""
    path = Path(os.path.abspath(os.fspath(path)))
    _root(path.parent)
    before = path.lstat()
    _need(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode),
          "FILE_SYMLINK_OR_NON_REGULAR")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    digest = hashlib.sha256()
    size = 0
    with os.fdopen(descriptor, "rb") as stream:
        _need(_identity(os.fstat(stream.fileno())) == _identity(before),
              "FILE_CHANGED_DURING_READ")
        while block := stream.read(CHUNK):
            digest.update(block)
            size += len(block)
        after = os.fstat(stream.fileno())
    _need(_identity(before) == _identity(after) == _identity(path.lstat())
          and size == after.st_size, "FILE_CHANGED_DURING_READ")
    return digest.hexdigest(), size


def _descriptor(row, *, exact=True):
    _need(isinstance(row, dict) and (not exact or set(row) == {"path", "sha256", "bytes"})
          and {"path", "sha256", "bytes"} <= set(row), "EXACT_FILE_DESCRIPTOR_REQUIRED")
    _relative(row["path"])
    _need(isinstance(row["sha256"], str) and re.fullmatch(r"[a-f0-9]{64}", row["sha256"])
          and type(row["bytes"]) is int and row["bytes"] >= 0,
          "INVALID_FILE_DESCRIPTOR")
    return row


def _matches(path, row):
    _need(hash_file(path) == (row["sha256"], row["bytes"]), "LOCAL_SHA_OR_BYTES_MISMATCH")


def _commit(root, relative, temporary, mode):
    """Link a verified owned temporary file, with no target replacement."""
    target = _file(root, relative, create_parents=True)
    _need(not target.exists(), "ATOMIC_TARGET_ALREADY_EXISTS")
    _root(temporary.parent)
    _need(target.parent == temporary.parent, "ATOMIC_PARENT_MISMATCH")
    _need(stat.S_ISREG(temporary.lstat().st_mode), "ATOMIC_TEMP_NOT_REGULAR")
    os.chmod(temporary, mode, follow_symlinks=False)
    directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            os.link(temporary.name, target.name, src_dir_fd=directory,
                    dst_dir_fd=directory, follow_symlinks=False)
        except FileExistsError:
            raise TransportError("ATOMIC_TARGET_ALREADY_EXISTS") from None
        os.fsync(directory)
    finally:
        os.close(directory)
    temporary.unlink()
    return target


def _temporary(parent):
    _root(parent)
    descriptor, name = tempfile.mkstemp(prefix=".paper3-", suffix=".partial", dir=parent)
    return os.fdopen(descriptor, "wb"), Path(name)


def _check_sdk_imports(site):
    for name, module in list(sys.modules.items()):
        root = name.partition(".")[0]
        if root not in SDK_IMPORT_ROOTS:
            continue
        source = getattr(module, "__file__", None)
        # six.moves deliberately creates synthetic submodules without files.
        if name != root and source is None:
            continue
        try:
            inside = source is not None and Path(source).resolve().is_relative_to(site)
        except (TypeError, ValueError, OSError):
            inside = False
        _need(inside, "LOADED_SDK_OUTSIDE_PINNED_SITE")


def ensure_sdk(bundle_dir, sdk_rows, work_dir):
    """Verify all eight pure Python wheels and expand them without pip/network."""
    global _sdk_site
    bundle = _root(bundle_dir)
    work = _root(work_dir, create=True)
    _need(isinstance(sdk_rows, list) and len(sdk_rows) == len(SDK_NAMES),
          "EXACT_EIGHT_SDK_WHEELS_REQUIRED")
    seen = set()
    wheels = []
    for row in sdk_rows:
        _descriptor(row)
        relative = _relative(row["path"])
        _need(len(relative.parts) == 2 and relative.parts[0] == "sdk"
              and re.fullmatch(r"[^/]+-(?:py3|py2\.py3)-none-any\.whl", relative.name),
              "PURE_PYTHON_SDK_WHEEL_REQUIRED")
        package = relative.name.split("-", 1)[0]
        _need(package in SDK_NAMES and package not in seen, "EXACT_EIGHT_SDK_WHEELS_REQUIRED")
        seen.add(package)
        wheel = _file(bundle, row["path"])
        _matches(wheel, row)
        wheels.append(wheel)
    _need(seen == SDK_NAMES, "EXACT_EIGHT_SDK_WHEELS_REQUIRED")
    site = _root(work / "sdk_site", create=True)
    _check_sdk_imports(site)
    member_names = set()
    try:
        for wheel in wheels:
            with zipfile.ZipFile(wheel) as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                _need(len(names) == len(set(names)), "SDK_DUPLICATE_ZIP_MEMBER")
                for info in infos:
                    name = info.filename.rstrip("/") if info.is_dir() else info.filename
                    _relative(name)
                    kind = stat.S_IFMT(info.external_attr >> 16)
                    _need(kind in {0, stat.S_IFREG, stat.S_IFDIR}
                          and not (info.flag_bits & 1), "SDK_LINK_OR_ENCRYPTED_MEMBER")
                    if info.is_dir():
                        _root(site / name, create=True)
                        continue
                    _need(kind != stat.S_IFDIR and name not in member_names,
                          "SDK_DUPLICATE_OR_NON_REGULAR_MEMBER")
                    member_names.add(name)
                    target = _file(site, name, create_parents=True)
                    digest = hashlib.sha256()
                    size = 0
                    output = temporary = None
                    try:
                        if not target.exists():
                            output, temporary = _temporary(target.parent)
                        with archive.open(info) as stream:
                            while block := stream.read(CHUNK):
                                size += len(block)
                                _need(size <= info.file_size, "SDK_MEMBER_LENGTH_MISMATCH")
                                digest.update(block)
                                if output is not None:
                                    output.write(block)
                        _need(size == info.file_size, "SDK_MEMBER_LENGTH_MISMATCH")
                        expected = {"sha256": digest.hexdigest(), "bytes": size}
                        if output is not None:
                            output.flush()
                            os.fsync(output.fileno())
                            output.close()
                            output = None
                            _commit(site, name, temporary, 0o644)
                            temporary = None
                        else:
                            _matches(target, expected)
                    finally:
                        if output is not None:
                            output.close()
                        if temporary is not None:
                            temporary.unlink(missing_ok=True)
    except TransportError:
        raise
    except Exception:
        raise TransportError("PINNED_SDK_EXTRACTION_FAILED") from None
    _sdk_site = site
    sys.dont_write_bytecode = True
    if str(site) not in sys.path:
        sys.path.insert(0, str(site))


def create_client(endpoint, bucket, access_key, secret_key):
    """Use normal SDK proxy/CA handling and explicit credentials only."""
    from urllib.parse import urlsplit
    _need(all(isinstance(value, str) and value and not any(ord(c) < 32 for c in value)
              for value in (endpoint, bucket, access_key, secret_key)), "R2_INPUT_REQUIRED")
    try:
        parsed = urlsplit(endpoint)
        endpoint_ok = (parsed.scheme == "https" and parsed.hostname is not None
                       and parsed.hostname.endswith(".r2.cloudflarestorage.com")
                       and parsed.username is None and parsed.password is None
                       and parsed.port in {None, 443} and parsed.path in {"", "/"}
                       and not parsed.query and not parsed.fragment)
    except ValueError:
        endpoint_ok = False
    _need(endpoint_ok, "R2_HTTPS_ENDPOINT_REQUIRED")
    _need(re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket) is not None,
          "R2_BUCKET_REQUIRED")
    # An embedding caller may otherwise have DEBUG logging enabled globally.
    for name in {"boto3", "botocore", "s3transfer", "urllib3"} | {
            name for name in logging.Logger.manager.loggerDict
            if name.startswith(("boto3.", "botocore.", "s3transfer.", "urllib3."))}:
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL + 1)
        logger.disabled = True
        logger.propagate = False
    try:
        _need(_sdk_site is not None, "PINNED_SDK_NOT_INITIALIZED")
        _check_sdk_imports(_sdk_site)
        import boto3
        from botocore.config import Config
        _check_sdk_imports(_sdk_site)
        client = boto3.client("s3", endpoint_url=endpoint.rstrip("/"), region_name="auto",
                              aws_access_key_id=access_key, aws_secret_access_key=secret_key,
                              config=Config(connect_timeout=15, read_timeout=120,
                                            retries={"max_attempts": 2, "mode": "standard"}))
        _check_sdk_imports(_sdk_site)
        return client
    except TransportError:
        raise
    except Exception:
        raise TransportError("R2_CLIENT_CREATION_FAILED") from None


def fetch_object(client, bucket, row, destination_root):
    """Complete one approved GET and atomically commit only exact bytes."""
    _descriptor(row, exact=False)
    _need(isinstance(row.get("key"), str) and row["key"].startswith("paper3/")
          and not any(ord(c) < 32 for c in row["key"]), "APPROVED_PAPER3_OBJECT_KEY_REQUIRED")
    root = _root(destination_root, create=True)
    target = _file(root, row["path"], create_parents=True)
    identity = {name: row[name] for name in ("path", "key", "sha256", "bytes")}
    if target.exists():
        _matches(target, row)
        return dict(identity, status="LOCAL_VERIFIED_CACHE_HIT", get_attempted=False,
                    http_status=None, complete_body_read=None, bytes_read=0,
                    get_sha256=None, local_sha256=row["sha256"], atomic_commit=None)
    body = output = temporary = None
    try:
        result = client.get_object(Bucket=bucket, Key=row["key"])
        body = result.get("Body")
        _need(result.get("ResponseMetadata", {}).get("HTTPStatusCode") == 200,
              "R2_GET_HTTP_NOT_200")
        declared = result.get("ContentLength")
        _need(type(declared) is int and declared == row["bytes"], "R2_GET_DECLARED_LENGTH_MISMATCH")
        _need(body is not None and callable(getattr(body, "read", None)), "R2_GET_BODY_MISSING")
        output, temporary = _temporary(target.parent)
        digest = hashlib.sha256()
        size = 0
        while block := body.read(CHUNK):
            _need(isinstance(block, bytes), "R2_GET_NON_BINARY_BODY")
            size += len(block)
            _need(size <= row["bytes"], "R2_GET_BODY_LENGTH_MISMATCH")
            digest.update(block)
            output.write(block)
        _need(size == row["bytes"] and digest.hexdigest() == row["sha256"],
              "R2_GET_BODY_SHA_OR_LENGTH_MISMATCH")
        output.flush()
        os.fsync(output.fileno())
        output.close()
        output = None
        target = _commit(root, row["path"], temporary, 0o600)
        temporary = None
        _matches(target, row)
        return dict(identity, status="R2_FULL_GET_SHA_VERIFIED", get_attempted=True,
                    http_status=200, complete_body_read=True, bytes_read=size,
                    declared_content_length=declared, get_sha256=digest.hexdigest(),
                    local_sha256=row["sha256"], atomic_commit="link_no_overwrite")
    except TransportError:
        raise
    except Exception:
        raise TransportError("R2_GET_FAILED") from None
    finally:
        if output is not None:
            output.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if body is not None:
            try:
                body.close()
            except Exception:
                pass


def probe_write(client, bucket, destination_root, release):
    """PUT a fresh harmless probe, then verify a complete GET of that object.

    The small object is deliberately retained; deletion permission is unneeded.
    This tests credentials and transport only and is never a research result.
    """
    _need(isinstance(release, dict)
          and isinstance(release.get("closure_sha256"), str)
          and re.fullmatch(r"[a-f0-9]{64}", release["closure_sha256"]),
          "WRITE_PROBE_RELEASE_CLOSURE_REQUIRED")
    closure = release["closure_sha256"]
    root = _root(destination_root, create=True)
    probe_id = uuid.uuid4().hex
    owned = root / ("probe_" + probe_id)
    try:
        owned.mkdir(mode=0o700)
    except FileExistsError:
        raise TransportError("WRITE_PROBE_OWNED_DIRECTORY_ALREADY_EXISTS") from None
    payload = (json.dumps({
        "schema_version": "paper3-r2-write-probe-v1",
        "purpose": "credential_write_read_preflight_only",
        "closure_sha256": closure,
        "scientific_result": False,
    }, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor = {"path": "probe.json", "bytes": len(payload),
                  "sha256": hashlib.sha256(payload).hexdigest(),
                  "key": "paper3/oneclick-probes/" + closure[:16] + "/" + probe_id + ".json"}
    try:
        response = client.put_object(Bucket=bucket, Key=descriptor["key"], Body=payload,
                                     ContentLength=len(payload), ContentType="application/json",
                                     IfNoneMatch="*")
        _need(response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 200,
              "R2_WRITE_PROBE_PUT_HTTP_NOT_200")
    except TransportError:
        raise
    except Exception:
        raise TransportError("R2_WRITE_PROBE_PUT_FAILED") from None
    try:
        verification = fetch_object(client, bucket, descriptor, owned)
        _need(verification["get_attempted"] is True
              and verification["status"] == "R2_FULL_GET_SHA_VERIFIED",
              "R2_WRITE_PROBE_FULL_GET_REQUIRED")
    except TransportError:
        raise TransportError("R2_WRITE_PROBE_FULL_GET_FAILED") from None
    return {"status": "R2_WRITE_AND_FULL_GET_PROBE_VERIFIED",
            "schema_version": "paper3-r2-write-probe-receipt-v1",
            "purpose": "credential_write_read_preflight_only",
            "closure_sha256": closure, "probe_object_key": descriptor["key"],
            "put_http_status": 200, "put_if_none_match": "*",
            "verification": verification, "probe_object_retained": True,
            "local_verified_path": str(owned / "probe.json"),
            "scientific_result": False, "training_started": False, "GPU_actions": 0}


def _package_inputs(path, expected_topdir):
    _need(len(_relative(expected_topdir).parts) == 1, "SINGLE_PACKAGE_TOPDIR_REQUIRED")
    expected_sha, expected_size = hash_file(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            before = _identity(os.fstat(stream.fileno()))
            raw = stream.read()
            _need(before == _identity(os.fstat(stream.fileno()))
                  == _identity(Path(path).lstat())
                  and (hashlib.sha256(raw).hexdigest(), len(raw))
                  == (expected_sha, expected_size), "FILE_CHANGED_DURING_READ")
        value = json.loads(raw)
    except TransportError:
        raise
    except Exception:
        raise TransportError("INVALID_EXACT_PACKAGE_INPUTS") from None
    _need(isinstance(value, dict)
          and value.get("schema_version") == "paper3-fixed-deployment-package-inputs-v1"
          and value.get("status") == "LOCAL_EXACT_PAYLOAD_LIST_NOT_AN_ARCHIVE_OR_UPLOAD"
          and value.get("top_level_directory") == expected_topdir
          and value.get("raw_EEG_embedded") is False
          and value.get("full_wheelhouse_embedded") is False,
          "FROZEN_PACKAGE_SCOPE_REQUIRED")
    rows = value.get("files")
    _need(isinstance(rows, list) and rows, "NONEMPTY_EXACT_PACKAGE_LIST_REQUIRED")
    expected = {}
    for row in rows:
        _descriptor(row)
        parts = _relative(row["path"]).parts
        _need(not any(part.lower().startswith(".env") or part.lower() in {
            ".git", ".ssh", ".aws", "credentials", "secrets"} for part in parts),
            "PRIVATE_FILE_IN_PACKAGE_FORBIDDEN")
        _need(row["path"] not in expected, "DUPLICATE_EXACT_PACKAGE_PATH")
        expected[row["path"]] = row
    _need(type(value.get("file_count")) is int and value["file_count"] == len(expected)
          and type(value.get("uncompressed_bytes")) is int
          and value["uncompressed_bytes"] == sum(row["bytes"] for row in rows),
          "EXACT_PACKAGE_COUNTERS_MISMATCH")
    return value, expected


def extract_exact(archive_path, input_path, root_dir, expected_topdir):
    """Merge into root_dir/expected_topdir after exact header/body checks.

    Existing identical files are only compared. Earlier individually verified
    members may remain after interruption; repeating the call verifies them.
    """
    _, rows = _package_inputs(input_path, expected_topdir)
    root = _root(root_dir, create=True)
    archive_path = Path(os.path.abspath(os.fspath(archive_path)))
    archive_hash, archive_bytes = hash_file(archive_path)
    before = _identity(archive_path.lstat())
    expected = {expected_topdir + "/" + path: row for path, row in rows.items()}
    copied = overlapped = total = 0
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            _need(len(names) == len(set(names)) and set(names) == set(expected),
                  "ARCHIVE_EXACT_MEMBER_SET_MISMATCH")
            for member in members:
                row = expected[member.name]
                mode = 0o755 if Path(row["path"]).suffix in {".py", ".sh"} else 0o644
                _need(member.type in {tarfile.REGTYPE, tarfile.AREGTYPE} and member.isfile()
                      and member.size == row["bytes"] and member.mode == mode
                      and member.uid == member.gid == 0 and member.mtime == 0
                      and not member.uname and not member.gname,
                      "ARCHIVE_FIXED_REGULAR_HEADER_REQUIRED")
                _relative(member.name)
            # tarfile ignores concatenated data after its first logical EOF.
            while trailer := archive.fileobj.read(CHUNK):
                _need(not any(trailer), "ARCHIVE_NONZERO_TRAILING_DATA")
            for member in members:
                row = expected[member.name]
                target = _file(root, member.name, create_parents=True)
                output = temporary = None
                digest = hashlib.sha256()
                size = 0
                try:
                    if not target.exists():
                        output, temporary = _temporary(target.parent)
                    stream = archive.extractfile(member)
                    _need(stream is not None, "ARCHIVE_MEMBER_BODY_MISSING")
                    with stream:
                        while block := stream.read(CHUNK):
                            size += len(block)
                            _need(size <= row["bytes"], "ARCHIVE_MEMBER_LENGTH_MISMATCH")
                            digest.update(block)
                            if output is not None:
                                output.write(block)
                    _need((digest.hexdigest(), size) == (row["sha256"], row["bytes"]),
                          "ARCHIVE_MEMBER_SHA_OR_LENGTH_MISMATCH")
                    if output is not None:
                        output.flush()
                        os.fsync(output.fileno())
                        output.close()
                        output = None
                        _commit(root, member.name, temporary, member.mode)
                        temporary = None
                        copied += 1
                    else:
                        _matches(target, row)
                        overlapped += 1
                    total += size
                finally:
                    if output is not None:
                        output.close()
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
        _need(before == _identity(archive_path.lstat()), "ARCHIVE_CHANGED_DURING_EXTRACTION")
    except TransportError:
        raise
    except Exception:
        raise TransportError("EXACT_ARCHIVE_EXTRACTION_FAILED") from None
    return {"status": "EXACT_ARCHIVE_ALL_MEMBERS_SHA_VERIFIED_AND_MERGED",
            "archive_sha256": archive_hash, "archive_bytes": archive_bytes,
            "verified_members": len(expected), "verified_payload_bytes": total,
            "new_atomic_copies": copied, "identical_existing_files": overlapped,
            "project_root": str(root / expected_topdir),
            "individual_member_network_GET_claimed": False,
            "training_started": False, "GPU_actions": 0}
