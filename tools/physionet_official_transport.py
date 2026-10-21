"""Unsigned official PhysioNet S3 transport and complete-file checksum oracle.

Only stdlib is imported. No environment, SDK, credentials or network is used at
import/catalog construction. Logical source IDs and canonical URLs never change.
GET suffix digests are explicitly not complete-file provider verification: the
native audit wrapper checks the original downloader's complete local-file SHA.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from types import MappingProxyType
import urllib.error
import urllib.request

SMALL_LIMIT = 16 * 1024**2
EVIDENCE_MANIFEST_PATH = "audit/source_schema/evidence_manifest.json"
PROVIDER_PATH = "audit/source_schema/evidence/physionet_provider.html"
CHECKSUMS_PATH = "audit/source_schema/evidence/physionet_SHA256SUMS.txt"
EVIDENCE_MANIFEST_SHA256 = "1aef5fd10e220c1614b0cbee26b5f2ad7b10bb81523175e866a95af265b987e4"
PROVIDER_SHA256 = "0f79d65a63535da798d2c202b3b0fbcde0f387adcdc9cf18537fc9f99cf8e052"
CHECKSUMS_SHA256 = "7f5d16957d8ee7bce86cc7ccba0e5994f63f33781607eb3f838392d49311a208"
OFFICIAL_COMMAND = "aws s3 sync --no-sign-request s3://physionet-open/eegmmidb/1.0.0/ DESTINATION"
CANONICAL_PREFIX = "https://physionet.org/files/eegmmidb/1.0.0/"
MIRROR_PREFIX = "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0/"
_HEADERS = ("Content-Length", "Content-Type", "Content-Range", "ETag", "Last-Modified", "Accept-Ranges")


class TransportError(ValueError):
    """Fixed public-data failure codes; never an exception's private payload."""


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _file_sha(path):
    result = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def _root(value):
    root = Path(value).absolute()
    if (root != root.resolve() or not root.is_dir()
            or any(p.is_symlink() for p in (root, *root.parents))):
        raise TransportError("official_transport_canonical_root_required")
    return root


def _local(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or relative.startswith("/") or any(x in {"", ".", ".."} for x in relative.split("/"))):
        raise TransportError("official_transport_safe_relative_path_required")
    result = root
    for part in relative.split("/"):
        result = result / part
        if result.is_symlink():
            raise TransportError("official_transport_symlinks_forbidden")
    return result


def _read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError):
        raise TransportError("official_transport_evidence_JSON_invalid") from None


def _descriptor(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": _file_sha(path), "bytes": path.stat().st_size}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".official-source-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _entry(subject, run):
    name = f"S{subject:03d}R{run:02d}.edf"
    relative = f"S{subject:03d}/{name}"
    ident = ["PhysionetMI", str(subject), "provider_single_visit", name]
    return {"dataset": ident[0], "provider_subject": ident[1], "session_id": ident[2],
        "file": name, "version": "1.0.0", "object_id": _digest(ident), "relative_path": relative,
        "url": CANONICAL_PREFIX + relative, "canonical_url": CANONICAL_PREFIX + relative,
        "mirror_url": MIRROR_PREFIX + relative}


class OfficialCatalog:
    """Fixed 327-file oracle, verified against all 22 retained evidence hashes."""
    __slots__ = ("_root", "_entries", "_evidence")

    @property
    def root(self):
        return self._root

    @property
    def entries(self):
        return self._entries

    @property
    def evidence(self):
        return self._evidence

    def __init__(self, root):
        self._root = _root(root)
        paths = {"manifest": EVIDENCE_MANIFEST_PATH, "provider": PROVIDER_PATH, "checksums": CHECKSUMS_PATH}
        pins = {"manifest": EVIDENCE_MANIFEST_SHA256, "provider": PROVIDER_SHA256, "checksums": CHECKSUMS_SHA256}
        evidence = {}
        for name, relative in paths.items():
            path = _local(self.root, relative)
            if not path.is_file() or _file_sha(path) != pins[name]:
                raise TransportError("original_official_transport_evidence_SHA_mismatch")
            evidence[name] = _descriptor(self.root, path)
        manifest = _read(_local(self.root, EVIDENCE_MANIFEST_PATH))
        files = manifest.get("files", [])
        if len(files) != 22 or len({f.get("path") for f in files}) != 22:
            raise TransportError("original_22_source_evidence_files_required")
        indexed = {}
        for item in files:
            path = _local(self.root, item.get("path"))
            if (not path.is_file() or _file_sha(path) != item.get("sha256")
                    or path.stat().st_size != item.get("bytes")):
                raise TransportError("original_22_source_evidence_integrity_mismatch")
            indexed[item["path"]] = item
        if (any(indexed.get(paths[n], {}).get("sha256") != pins[n] for n in ("provider", "checksums"))
                or indexed[PROVIDER_PATH].get("source_url_or_derivation") != "https://physionet.org/content/eegmmidb/1.0.0/"
                or indexed[CHECKSUMS_PATH].get("source_url_or_derivation") != CANONICAL_PREFIX + "SHA256SUMS.txt"):
            raise TransportError("official_provider_and_checksum_source_identity_required")
        lines = _local(self.root, PROVIDER_PATH).read_text().splitlines()
        if len(lines) < 746 or "Download the files using AWS" not in lines[744] or OFFICIAL_COMMAND not in lines[745]:
            raise TransportError("retained_official_unsigned_S3_instruction_required")
        sums = {}
        for line in _local(self.root, CHECKSUMS_PATH).read_text().splitlines():
            parts = line.split()
            if (len(parts) != 2 or re.fullmatch(r"[0-9a-f]{64}", parts[0]) is None
                    or parts[1].startswith("/") or "\\" in parts[1]
                    or any(p in {"", ".", ".."} for p in parts[1].split("/")) or parts[1] in sums):
                raise TransportError("official_checksum_catalog_has_unsafe_or_duplicate_entries")
            sums[parts[1]] = parts[0]
        entries = {}
        for subject in range(1, 110):
            for run in (4, 8, 12):
                value = _entry(subject, run)
                if value["relative_path"] not in sums:
                    raise TransportError("all_327_fixed_source_EDFs_need_official_checksums")
                value["sha256"] = sums[value["relative_path"]]
                entries[value["relative_path"]] = MappingProxyType(value)
        self._entries = MappingProxyType(entries)
        self._evidence = MappingProxyType({k: MappingProxyType(v) for k, v in evidence.items()})

    def lookup(self, entry):
        subject = entry.get("provider_subject")
        if not isinstance(subject, str) or not subject.isdigit() or not 1 <= int(subject) <= 109:
            raise TransportError("exact_327_canonical_source_identity_required")
        relative = f"S{int(subject):03d}/{entry.get('file', '')}"
        expected = self.entries.get(relative)
        fields = ("dataset", "provider_subject", "session_id", "file", "version", "object_id", "url")
        if expected is None or any(entry.get(k) != expected[k] for k in fields):
            raise TransportError("exact_327_canonical_source_identity_required")
        return dict(expected)


def load_official_catalog(root):
    return OfficialCatalog(root)


def _checksum_proof(entry, actual_sha256, catalog):
    expected = catalog.lookup(entry)
    if actual_sha256 != expected["sha256"]:
        raise TransportError("complete_raw_SHA256_does_not_match_original_official_checksum")
    return {"schema_version": "paper3-official-PhysionetMI-complete-file-checksum-v1",
        "provider_sha256_verified": True, "verification_scope": "complete_original_file_not_ETag_or_range_suffix",
        "canonical_url": expected["canonical_url"], "relative_object": expected["relative_path"],
        "expected_sha256": expected["sha256"], "actual_complete_file_sha256": actual_sha256,
        "official_checksum_catalog": dict(catalog.evidence["checksums"])}


def verify_completed_provider_checksum(entry, state, catalog):
    """Compare a historical full-file digest; caller also verifies its R2 proof.

    No historical COMPLETE state or claimed transport is changed. This function
    alone is not an R2 verifier or a grant of source/W2 completion capability.
    """
    expected = catalog.lookup(entry)
    if (state.get("status") != "COMPLETE" or state.get("identity") != [expected[k] for k in
            ("dataset", "provider_subject", "session_id", "file")]
            or state.get("raw_integrity", {}).get("complete_body_read") is not True):
        raise TransportError("completed_original_file_digest_provenance_required")
    return _checksum_proof(entry, state["raw_integrity"].get("sha256"), catalog)


def verify_all_completed_provider_checksums(root, plan, batch_dir, catalog):
    """Mandatory aggregate/return preflight, including old canonical COMPLETEs."""
    root, batch_dir = _root(root), Path(batch_dir)
    if catalog.root != root or not batch_dir.is_relative_to(root):
        raise TransportError("official_checksum_scan_root_mismatch")
    entries = [v for v in plan["files"] if v.get("dataset") == "PhysionetMI"]
    if len(entries) != 327 or {catalog.lookup(e)["relative_path"] for e in entries} != set(catalog.entries):
        raise TransportError("official_checksum_scan_requires_exact_327_source_selection")
    verified = 0
    for entry in entries:
        path = _local(root, (batch_dir / "objects" / entry["object_id"] / "state.json").relative_to(root).as_posix())
        if path.is_file():
            state = _read(path)
            if state.get("status") == "COMPLETE":
                verify_completed_provider_checksum(entry, state, catalog)
                verified += 1
    return {"status": "EXISTING_COMPLETE_SOURCE_CHECKSUMS_MATCH", "verified_complete_files": verified,
        "required_source_files": 327, "source_or_W2_completion_granted": False}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def _unsigned_open(request, *, timeout):
    opener = urllib.request.build_opener(_NoRedirect())
    return opener.open(request, timeout=timeout)


class OfficialPhysionetTransport:
    """Exact unsigned mapping, framed bodies and independent private receipts."""
    @property
    def entry(self):
        return self._entry

    @property
    def head(self):
        return self._head

    @property
    def limit(self):
        return self._limit

    def __init__(self, root, entry, *, catalog=None, max_bytes=SMALL_LIMIT):
        self.root = _root(root)
        if type(max_bytes) is not int or not 1 <= max_bytes <= SMALL_LIMIT:
            raise TransportError("official_transport_limit_must_not_exceed_16MiB")
        self.catalog = catalog if catalog is not None else load_official_catalog(self.root)
        if not isinstance(self.catalog, OfficialCatalog) or self.catalog.root != self.root:
            raise TransportError("same_root_verified_official_catalog_required")
        self._entry, self._limit, self._head = MappingProxyType(self.catalog.lookup(entry)), max_bytes, None
        self._lock, self._events, self._provider = threading.RLock(), [], None
        self._id = _digest({"object_id": self.entry["object_id"], "utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()})
        relative = f"audit/source_preparation/official_transport/{self.entry['object_id']}/{self._id}.json"
        self.receipt_path = _local(self.root, relative)
        if self.receipt_path.exists():
            raise TransportError("official_transport_receipt_identity_collision")

    def receipt(self):
        with self._lock:
            complete_GET = any(e.get("method") == "GET" and e.get("complete_body_read")
                and e.get("resumed_from_bytes") == 0 for e in self._events)
            suffix_GET = any(e.get("method") == "GET" and e.get("complete_body_read") for e in self._events)
            value = {"schema_version": "paper3-official-PhysionetMI-transport-v1", "transport": "official-s3",
                "dataset": "PhysionetMI", "source_version": "1.0.0", "object_id": self.entry["object_id"],
                "canonical_url": self.entry["canonical_url"], "mirror_url": self.entry["mirror_url"],
                "expected_sha256": self.entry["sha256"], "relative_object": self.entry["relative_path"],
                "official_evidence": {k: dict(v) for k, v in self.catalog.evidence.items()},
                "unsigned_origin_requests": True, "SDK_or_R2_credentials_used": False,
                "raw_body_transport": "official-s3-complete-file-GET-observed" if complete_GET else
                    "official-s3-suffix-GET-only-prefix-origin-not-inferred" if suffix_GET else "no_complete_mirror_GET_claim",
                "provider_complete_file_checksum": json.loads(json.dumps(self._provider)),
                "events": json.loads(json.dumps(self._events))}
            value["artifact_sha256"] = _digest(value)
            return value

    def _persist(self):
        with self._lock:
            _local(self.root, self.receipt_path.relative_to(self.root).as_posix())
            _write(self.receipt_path, self.receipt())

    def __call__(self, request, *, timeout=90):
        method = request.get_method()
        incoming = {k.lower(): v for k, v in request.header_items()}
        if (request.full_url != self.entry["canonical_url"] or method not in {"HEAD", "GET"}
                or request.data is not None or set(incoming) - {"if-match", "range"}
                or method == "HEAD" and incoming):
            raise TransportError("unsigned_exact_canonical_source_request_required")
        offset = 0
        if method == "GET":
            if self.head is None or incoming.get("if-match") != self.head["etag"]:
                raise TransportError("official_GET_requires_matching_fresh_HEAD_and_IfMatch")
            if "range" in incoming:
                match = re.fullmatch(r"bytes=([1-9][0-9]*)-", incoming["range"])
                if match is None:
                    raise TransportError("official_GET_requires_exact_open_ended_Range")
                offset = int(match.group(1))
            if offset >= self.head["bytes"]:
                raise TransportError("official_GET_range_must_be_inside_verified_total")
        outgoing = urllib.request.Request(self.entry["mirror_url"], method=method,
            headers={k: v for k, v in request.header_items()})
        event = {"method": method, "canonical_url": request.full_url,
            "requested_url": self.entry["mirror_url"], "final_url": None,
            "request_headers": dict(request.header_items()), "response_headers": {},
            "status": "REQUEST_INTENT", "bytes_read": 0, "complete_body_read": False,
            "expected_original_checksum": {"catalog": dict(self.catalog.evidence["checksums"]),
                "entry": self.entry["relative_path"], "sha256": self.entry["sha256"]}}
        with self._lock:
            self._events.append(event)
            self._persist()
        response = None
        try:
            response = _unsigned_open(outgoing, timeout=timeout)
            event["final_url"] = response.geturl()
            event["http_status"] = response.status
            event["response_headers"] = {k: response.headers[k] for k in _HEADERS if response.headers.get(k) is not None}
            if event["final_url"] != self.entry["mirror_url"]:
                raise TransportError("official_mirror_redirect_or_final_URL_change_forbidden")
            length = response.headers.get("Content-Length")
            if not isinstance(length, str) or re.fullmatch(r"[1-9][0-9]*", length) is None or int(length) > self.limit:
                raise TransportError("official_mirror_positive_bounded_ContentLength_required")
            etag, modified = response.headers.get("ETag"), response.headers.get("Last-Modified")
            if not isinstance(etag, str) or not etag or not isinstance(modified, str) or not modified:
                raise TransportError("official_mirror_HEAD_GET_validators_required")
            if method == "HEAD":
                if response.status != 200:
                    raise TransportError("official_mirror_HEAD_requires_200")
                self._head = MappingProxyType({"bytes": int(length), "etag": etag, "modified": modified})
                event["status"] = "HEAD_VERIFIED_METADATA_ONLY"
                self._persist()
                return response
            if (response.status != (206 if offset else 200) or int(length) != self.head["bytes"] - offset
                    or etag != self.head["etag"] or modified != self.head["modified"]
                    or offset and response.headers.get("Content-Range") != f"bytes {offset}-{self.head['bytes']-1}/{self.head['bytes']}"):
                raise TransportError("official_mirror_GET_frame_or_fresh_validators_mismatch")
            event.update(status="GET_OPEN_FRAME_VERIFIED", resumed_from_bytes=offset,
                body_digest_scope="range_suffix_only_not_complete_file" if offset else "complete_GET_body_not_yet_provider_verified")
            self._persist()
            return _Body(response, self, event, int(length))
        except Exception as error:
            if isinstance(error, urllib.error.HTTPError):
                event["http_status"] = error.code
                event["final_url"] = error.geturl()
                event["response_headers"] = {k: error.headers[k] for k in _HEADERS if error.headers.get(k) is not None}
                event["redirect_response_not_followed"] = 300 <= error.code < 400
                error.close()
            if response is not None:
                response.close()
            event.update(status="REQUEST_FAILED_CLOSED", failure_type=type(error).__name__)
            self._persist()
            if isinstance(error, TransportError):
                raise
            raise TransportError("official_mirror_request_failed_details_suppressed") from None

    def finalize_complete_file(self, actual_sha256):
        self._provider = _checksum_proof(self.entry, actual_sha256, self.catalog)
        self._persist()
        return {"receipt": _descriptor(self.root, self.receipt_path), "proof": self.receipt()}


class _Body:
    def __init__(self, response, owner, event, length):
        self.response, self.owner, self.event, self.length = response, owner, event, length
        self.headers, self.status, self.hash, self.closed = response.headers, response.status, sha256(), False

    def geturl(self):
        return self.response.geturl()

    def read(self, amount=-1):
        if self.closed or type(amount) is not int or amount < -1:
            raise TransportError("official_mirror_body_read_parameters_invalid")
        if amount == 0:
            return b""
        remaining = self.length - self.event["bytes_read"]
        if remaining < 0:
            raise TransportError("official_mirror_body_frame_already_failed")
        requested = remaining + 1 if amount == -1 else min(amount, remaining + 1)
        try:
            block = self.response.read(requested)
        except Exception:
            self.event["status"] = "GET_BODY_READ_FAILED"
            self.owner._persist()
            raise TransportError("official_mirror_body_read_failed_details_suppressed") from None
        if not isinstance(block, bytes):
            raise TransportError("official_mirror_body_must_be_binary")
        self.event["bytes_read"] += len(block)
        self.hash.update(block)
        self.event["body_digest_sha256"] = self.hash.hexdigest()
        if self.event["bytes_read"] > self.length:
            self.event["status"] = "GET_BODY_FRAME_FAILED"
            self.owner._persist()
            # The original downloader must observe this bounded probe before
            # its existing frame guard raises: its durable consume records all
            # actual bytes once. The full reservation already includes the probe.
            return block
        if not block and self.event["bytes_read"] != self.length:
            self.event["status"] = "GET_BODY_FRAME_FAILED"
            self.owner._persist()
            raise TransportError("official_mirror_body_overrun_or_incomplete_EOF")
        if not block:
            self.event.update(status="GET_COMPLETE_FRAME_ONLY", complete_body_read=True)
        self.owner._persist()
        return block

    def close(self):
        if not self.closed:
            self.closed = True
            if not self.event["complete_body_read"] and not self.event["status"].endswith("FAILED"):
                self.event["status"] = "GET_CLOSED_WITHOUT_COMPLETE_EOF"
            self.response.close()
            self.owner._persist()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ProviderChecksumAudit:
    """Compare the downloader's complete-file SHA before native EEG parsing.

    A mismatch raises inside the original native-audit stage: its established
    quarantine path preserves raw/evidence and can never mark the object COMPLETE.
    No fourth object is added to the original three-object R2 manifest.
    """
    def __init__(self, inner_audit, catalog, transport_lookup=None):
        self.inner, self.catalog, self.transport_lookup = inner_audit, catalog, transport_lookup

    def __call__(self, dataset, path, **kwargs):
        if dataset != "PhysionetMI":
            return self.inner(dataset, path, **kwargs)
        try:
            subject = int(kwargs["provider_subject"])
            match = re.fullmatch(rf"S{subject:03d}R(04|08|12)\.edf", Path(path).name)
            if match is None or kwargs.get("source_version") != "1.0.0" or kwargs.get("session_id") != "provider_single_visit":
                raise TransportError("official_checksum_audit_native_identity_mismatch")
            entry = _entry(subject, int(match.group(1)))
            proof = _checksum_proof(entry, kwargs.get("expected_sha256"), self.catalog)
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, TransportError):
                raise
            raise TransportError("official_checksum_audit_native_identity_invalid") from None
        transport = self.transport_lookup(entry) if self.transport_lookup is not None else None
        if transport is not None and (not isinstance(transport, OfficialPhysionetTransport)
                or transport.entry["object_id"] != entry["object_id"] or transport.catalog.root != self.catalog.root):
            raise TransportError("checksum_audit_transport_must_match_exact_original_object")
        transport_proof = transport.finalize_complete_file(kwargs["expected_sha256"]) if transport is not None else None
        result = self.inner(dataset, path, **kwargs)
        result["summary"]["official_provider_checksum"] = proof
        if transport_proof is not None:
            result["summary"]["official_source_transport"] = transport_proof
        return result
