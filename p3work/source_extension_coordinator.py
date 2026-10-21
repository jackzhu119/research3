"""Serial, restartable CPU preparation of the two frozen source extensions.

This coordinator owns only its explicit raw staging files. It never opens an
external cohort, fits a model, starts a GPU, resets the separate W1 ledger, or
deletes an existing unowned pilot. Every raw and cache object needs complete
R2 GET/SHA evidence before an owned raw file can be removed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
import time
from urllib.parse import urlsplit
import zlib

import numpy as np

from p3work import source_extension_io as storage

SCHEMA = "paper3-source-extension-cache-v1"
CONTRACT_SHA = "5c8ea3f01634c060546af29b9330d314c4f03c265677a74b875ac7c302e31872"
CHANNELS = ["FC3", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP4"]
STAGE = "pre_rms_car11_temporal_demeaned"
FROZEN_PROFILES = {
    "Yang2025": "af1e59825c1bb1e78e6eaa2115579a4b0f14245cb6ed3c280df4233184cc17eb",
    "Dreyer2023": "7f7a35c3015f68dc6d6caee042d3b822b37c20c54a85c1814c90785b8e9dbf48",
}
EXPECTED_COUNTS = {"Yang2025": 153, "Dreyer2023": 520}
ADAPTERS = {"Yang2025": "p3work.source_extensions.yang2025",
            "Dreyer2023": "p3work.source_extensions.dreyer2023"}
RANGE_CHUNK = 8 * 1024**2
CHUNK = 1024**2
CREATOR = "tools/prepare_source_extensions.py"


class PreparationError(ValueError):
    """A fixed public error code, never an SDK or provider error message."""


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha_file(path: Path) -> tuple[str, int]:
    h, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def verify_raw_file(path: Path, item: dict) -> dict:
    h, size, crc = hashlib.sha256(), 0, 0
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK):
            h.update(chunk)
            size += len(chunk)
            crc = zlib.crc32(chunk, crc)
    actual = {"sha256": h.hexdigest(), "bytes": size, "crc32": f"{crc & 0xffffffff:08x}"}
    if (actual["sha256"] != item["sha256"] or actual["bytes"] != item["bytes"]
            or ("crc32" in item and actual["crc32"] != item["crc32"])):
        raise PreparationError("original_native_sha_size_or_crc_mismatch")
    return actual


def local(root: Path, relative: str) -> Path:
    if (not isinstance(relative, str) or not relative or relative.startswith("/")
            or "\\" in relative or any(p in {"", ".", ".."} for p in relative.split("/"))
            or any(ord(c) < 32 for c in relative)):
        raise PreparationError("unsafe_relative_path")
    path = root / relative
    storage._credential_components(path)
    cursor = Path(os.path.abspath(path))
    for part in (cursor, *cursor.parents):
        if part.is_symlink():
            raise PreparationError("symlink_forbidden")
    if not path.resolve().is_relative_to(root.resolve()):
        raise PreparationError("path_outside_root")
    return path


def descriptor(root: Path, path: Path) -> dict:
    path = local(root, path.relative_to(root).as_posix())
    sha, size = sha_file(path)
    return {"path": path.relative_to(root).as_posix(), "sha256": sha, "bytes": size}


def checked(root: Path, item: dict) -> Path:
    if (not isinstance(item, dict) or not isinstance(item.get("path"), str)
            or not re.fullmatch("[0-9a-f]{64}", str(item.get("sha256", "")))):
        raise PreparationError("explicit_artifact_sha256_required")
    path = local(root, item["path"])
    actual = descriptor(root, path)
    if actual["sha256"] != item["sha256"] or ("bytes" in item and actual["bytes"] != item["bytes"]):
        raise PreparationError("artifact_sha_or_bytes_mismatch")
    return path


def _sync_parent(path: Path) -> None:
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value, *, immutable=False) -> None:
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise PreparationError("symlink_forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise PreparationError("symlink_forbidden")
    if immutable and path.exists():
        if json.loads(path.read_bytes()) != value:
            raise PreparationError("existing_immutable_artifact_differs")
        return
    fd, name = tempfile.mkstemp(prefix=".source-extension-", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        _sync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def array_once(path: Path, array: np.ndarray) -> None:
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise PreparationError("symlink_forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".source-extension-", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            if sha_file(path) != sha_file(temporary):
                raise PreparationError("existing_array_differs")
        else:
            os.link(temporary, path)
            _sync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def safe_url(url: str) -> str:
    p = urlsplit(url)
    if (p.scheme != "https" or not p.hostname or p.username or p.password or p.fragment or p.query
            or p.hostname not in {"data.nemar.org", "ndownloader.figshare.com"}):
        raise PreparationError("unapproved_or_credential_bearing_provider_url")
    return url


def normalize_inventory(dataset: str, inventory: dict) -> list[dict]:
    if dataset == "Dreyer2023":
        if inventory.get("schema_version") != "paper3-dreyer-source-object-plan-v1":
            raise PreparationError("unexpected_official_inventory_schema")
        result = []
        for row in inventory["objects"]:
            relative = row["native_relative_path"]
            subject = f"sub-{int(row['provider_subject_id']):02d}"
            run = int(row["run_index"])
            name = f"{subject}/eeg/{subject}_task-R{run}{'acquisition' if run <= 2 else 'online'}_eeg.edf"
            expected_url = "https://data.nemar.org/nm000250/v1.0.5/sourcedata/" + name
            if (row["dataset_id"] != dataset or not 1 <= run <= 6 or relative != name
                    or row["url"] != expected_url or row["manifest_path"] != "sourcedata/"+name):
                raise PreparationError("official_identity_or_run_invalid")
            component = {"path": f"{storage.RAW_ROOT}/{dataset}/{relative}", "bytes": row["bytes"],
                "sha256": row["sha256"], "download": {"kind": "direct", "url": safe_url(row["url"])}}
            result.append({"record_id": f"{subject}_run-{run}", "subject_id": subject,
                           "session_id": "provider_single_day", "raw_files": [component]})
    elif dataset == "Yang2025":
        if (inventory.get("schema_version") != "paper3-yang-source-download-selection-v1"
                or inventory.get("current_all312_inventory_fields_verified") is not True
                or inventory.get("archive_url") != "https://ndownloader.figshare.com/files/51001884"
                or inventory.get("archive_bytes") != 65630556299
                or inventory.get("current_central_directory_sha256") !=
                "cecd7ea0895da6ef85f01405960f4844c42d788ad33186b1f3fca91744d6874a"):
            raise PreparationError("frozen_complete_yang_inventory_required")
        pairs = {}
        for row in inventory["members"]:
            match = re.fullmatch(r"WBCIC_SHU Motor Imagery dataset/sourcedata/2C dataset/"
                                 r"(sub-\d{3})/(ses-\d{2})/eeg/(data|evt)\.bdf", row["path"])
            if not match:
                raise PreparationError("yang_selection_contains_non2C_member")
            subject, session, name = match.groups()
            key = (subject, session)
            component = {"path": f"{storage.RAW_ROOT}/{dataset}/{subject}/{session}/{name}.bdf",
                "bytes": row["bytes"], "sha256": row["expected_native_sha256"], "crc32": row["crc32"],
                "download": {"kind": "zip_range", "url": safe_url(inventory["archive_url"]),
                    "archive_bytes": inventory["archive_bytes"], "member": row["path"],
                    "header_offset": row["header_offset"], "compressed_bytes": row["compressed_bytes"],
                    "compression": row["compression"],
                    "central_directory_sha256": inventory["current_central_directory_sha256"]}}
            pairs.setdefault(key, []).append(component)
        result = [{"record_id": subject + "_" + session, "subject_id": subject,
                   "session_id": session, "raw_files": sorted(rows, key=lambda r: r["path"])}
                  for (subject, session), rows in sorted(pairs.items())]
        if any({Path(f["path"]).name for f in row["raw_files"]} != {"data.bdf", "evt.bdf"} for row in result):
            raise PreparationError("exact_data_evt_pair_required")
    else:
        raise PreparationError("dataset_not_in_frozen_clinic11_source_branch")
    if len(result) != EXPECTED_COUNTS[dataset] or len({r["record_id"] for r in result}) != len(result):
        raise PreparationError("all_frozen_records_required")
    observed = {(r["subject_id"], r["session_id"] if dataset == "Yang2025" else int(r["record_id"].rsplit("-", 1)[1]))
                for r in result}
    expected = ({(f"sub-{s:03d}", f"ses-{r:02d}") for s in range(1, 52) for r in range(1, 4)}
                if dataset == "Yang2025" else {(f"sub-{s:02d}", r) for s in range(1, 88)
                                               for r in range(1, 7) if not (s == 59 and r in {5, 6})})
    if observed != expected:
        raise PreparationError("exact_frozen_provider_subject_and_record_membership_required")
    paths = set()
    for row in result:
        for item in row["raw_files"]:
            storage.validate_entry({"path": item["path"], "sha256": item["sha256"], "bytes": item["bytes"],
                                    "role": "source_development_raw", "dataset": dataset})
            if item["bytes"] <= 0 or item["path"] in paths:
                raise PreparationError("nonpositive_or_duplicate_raw_component")
            spec = item["download"]
            if spec["kind"] == "zip_range" and (
                    type(spec["header_offset"]) is not int or spec["header_offset"] < 0
                    or type(spec["compressed_bytes"]) is not int or spec["compressed_bytes"] <= 0
                    or spec["header_offset"] + spec["compressed_bytes"] + 30 > spec["archive_bytes"]
                    or spec["compression"] != 8 or re.fullmatch("[0-9a-f]{8}", item["crc32"]) is None):
                raise PreparationError("invalid_frozen_zip_member_integrity_or_range")
            paths.add(item["path"])
    return result


def build_plan(root: Path, *, dataset: str, inventory_path: Path, profile_path: Path,
               max_provider_body_bytes=None, max_r2_get_body_bytes=None,
               max_wall_seconds=172800, max_cache_bytes=4 * 1024**3, disk_reserve_bytes=2 * 1024**3) -> dict:
    root = Path(os.path.abspath(root))
    local(root, "audit")
    if dataset not in FROZEN_PROFILES:
        raise PreparationError("dataset_not_in_frozen_clinic11_source_branch")
    profile_descriptor = descriptor(root, profile_path)
    if profile_descriptor["sha256"] != FROZEN_PROFILES[dataset]:
        raise PreparationError("source_profile_not_exact_root_frozen_document")
    native_profile = json.loads(profile_path.read_bytes())
    contract_sha = native_profile.get("input_contract_sha256", native_profile.get("signal_contract_semantic_sha256"))
    if contract_sha != CONTRACT_SHA:
        raise PreparationError("frozen_primary11_contract_required")
    records = normalize_inventory(dataset, json.loads(inventory_path.read_bytes()))
    code_names = ["p3work/source_extension_coordinator.py", "tools/prepare_source_extensions.py",
                  "p3work/source_extension_io.py", "tools/r2_backup.py", "tools/r2_restore.py",
                  "p3work/preprocess.py", "p3work/contracts.py", ADAPTERS[dataset].replace(".", "/") + ".py"]
    codes = {name: descriptor(root, local(root, name)) for name in code_names}
    historical = []
    candidate = ("audit/data_plan_20261019/yang/single_session_download_receipt.json" if dataset == "Yang2025"
                 else "audit/data_plan_20261019/dreyer/pilot/raw_GET_receipt.json")
    if local(root, candidate).is_file():
        historical.append(descriptor(root, local(root, candidate)))
    raw_bytes = sum(f["bytes"] for r in records for f in r["raw_files"])
    provider_bytes = sum(f["download"].get("compressed_bytes", f["bytes"]) +
        (131100 if f["download"]["kind"] == "zip_range" else 0) for r in records for f in r["raw_files"])
    budgets = {"provider_body_bytes": max_provider_body_bytes if max_provider_body_bytes is not None else 2 * provider_bytes,
               "r2_get_body_bytes": max_r2_get_body_bytes if max_r2_get_body_bytes is not None else 2 * raw_bytes + 2 * max_cache_bytes,
               "wall_seconds": max_wall_seconds, "cache_bytes": max_cache_bytes, "disk_reserve_bytes": disk_reserve_bytes}
    if any(type(v) is not int or v <= 0 for v in budgets.values()):
        raise PreparationError("positive_integer_resource_budgets_required")
    plan = {"schema_version": "paper3-source-extension-plan-v1", "dataset": dataset,
        "role": "source_train", "task_subset": "2C" if dataset == "Yang2025" else "LR",
        "document_sha256": storage.DOCUMENT_SHA256, "source_only": True,
        "inventory": descriptor(root, inventory_path), "native_profile": profile_descriptor,
        "code": codes, "input_contract_sha256": CONTRACT_SHA, "channels": CHANNELS,
        "records": records, "record_count": len(records), "raw_uncompressed_bytes": raw_bytes,
        "historical_get_evidence": historical,
        "budget_scope": "new coordinator body reservations only; inherited GET proofs retained separately; no previous ledger reset",
        "wall_accounting": "cumulative active runtime plus downtime of any unclosed durable clock intent",
        "budgets": budgets, "inflight_records": 1, "GPU_actions": 0, "fits": 0,
        "external_payload_access_allowed": False, "old_w1_ledger_modified": False}
    plan["plan_sha256"] = digest(plan)
    return plan


def validate_plan(root: Path, plan: dict) -> dict:
    copy = dict(plan)
    sha = copy.pop("plan_sha256", None)
    if digest(copy) != sha:
        raise PreparationError("plan_sha256_mismatch")
    for item in [plan["inventory"], plan["native_profile"], *plan["code"].values()]:
        checked(root, item)
    expected = build_plan(root, dataset=plan["dataset"], inventory_path=checked(root, plan["inventory"]),
        profile_path=checked(root, plan["native_profile"]),
        max_provider_body_bytes=plan["budgets"]["provider_body_bytes"],
        max_r2_get_body_bytes=plan["budgets"]["r2_get_body_bytes"],
        max_wall_seconds=plan["budgets"]["wall_seconds"], max_cache_bytes=plan["budgets"]["cache_bytes"],
        disk_reserve_bytes=plan["budgets"]["disk_reserve_bytes"])
    if expected != plan:
        raise PreparationError("frozen_selection_profile_code_or_budget_changed")
    return json.loads(checked(root, plan["native_profile"]).read_bytes())


def clock_snapshot() -> dict:
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if not re.fullmatch("[0-9a-f-]{36}", boot):
        raise PreparationError("boot_clock_unavailable")
    return {"utc": datetime.now(timezone.utc).isoformat(), "monotonic_ns": time.monotonic_ns(), "boot_id": boot}


def elapsed(start: dict, now: dict) -> float:
    try:
        a, b = datetime.fromisoformat(start["utc"]), datetime.fromisoformat(now["utc"])
        wall, mono = (b-a).total_seconds(), (now["monotonic_ns"]-start["monotonic_ns"])/1e9
        if (a.tzinfo is None or b.tzinfo is None or start["boot_id"] != now["boot_id"]
                or not all(math.isfinite(v) and v >= 0 for v in (wall, mono)) or abs(wall-mono) > 5):
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError):
        raise PreparationError("elapsed_unknown_or_clock_changed_budget_closed") from None
    return max(wall, mono)


class Ledger:
    def __init__(self, path: Path, plan: dict):
        self.path, self.plan = path, plan
        self.data = json.loads(path.read_bytes()) if path.exists() else {
            "schema_version": "paper3-source-extension-budget-ledger-v1", "plan_sha256": plan["plan_sha256"],
            "charged": {"provider_body_bytes": 0, "r2_get_body_bytes": 0}, "elapsed_seconds": 0., "history": []}
        if self.data.get("plan_sha256") != plan["plan_sha256"]:
            raise PreparationError("existing_budget_ledger_plan_mismatch")
        self.checkpoint()
        if self.data.get("active_transfer"):
            self.data["history"].append(dict(self.data.pop("active_transfer"),
                status="HARD_INTERRUPTION_FULL_RESERVATION_CONSUMED"))
            self.save()

    def save(self):
        atomic_json(self.path, self.data)

    def checkpoint(self, *, close=False):
        now = clock_snapshot()
        if self.data.get("wall_intent"):
            self.data["elapsed_seconds"] += elapsed(self.data["wall_intent"], now)
        self.data.pop("wall_intent", None)
        if not close:
            self.data["wall_intent"] = now
        self.save()
        if self.data["elapsed_seconds"] > self.plan["budgets"]["wall_seconds"]:
            raise PreparationError("cumulative_wall_budget_exhausted")

    def reserve(self, kind: str, size: int, identity: str):
        self.checkpoint()
        if self.data.get("active_transfer"):
            raise PreparationError("unclosed_transfer_reservation")
        if type(size) is not int or size < 0 or kind not in self.data["charged"]:
            raise PreparationError("invalid_transfer_reservation")
        if self.data["charged"][kind] + size > self.plan["budgets"][kind]:
            raise PreparationError("cumulative_body_budget_exhausted")
        self.data["charged"][kind] += size
        self.data["active_transfer"] = {"kind": kind, "reserved_bytes": size, "identity": identity}
        self.save()  # This must precede the provider/R2 request.

    def finish(self, *, verified_bytes=None):
        transfer = self.data.pop("active_transfer", None)
        if transfer is None:
            return
        if verified_bytes is not None:
            if type(verified_bytes) is not int or not 0 <= verified_bytes <= transfer["reserved_bytes"]:
                self.data["active_transfer"] = transfer
                self.save()
                raise PreparationError("successful_body_exceeded_reservation")
            self.data["charged"][transfer["kind"]] += verified_bytes-transfer["reserved_bytes"]
        self.data["history"].append(dict(transfer, verified_bytes=verified_bytes,
            status="SUCCESS_FULL_BODY_VERIFIED" if verified_bytes is not None else "FAILURE_FULL_RESERVATION_CONSUMED"))
        self.checkpoint()


def _range(url: str, start: int, length: int, archive_bytes: int, *, session, ledger: Ledger) -> bytes:
    ledger.reserve("provider_body_bytes", length, f"range:{start}:{length}")
    response = None
    try:
        response = session.get(url, headers={"Range": f"bytes={start}-{start+length-1}",
            "Accept-Encoding": "identity"}, stream=True, timeout=(20, 120))
        if (response.status_code != 206 or response.headers.get("Content-Range") !=
                f"bytes {start}-{start+length-1}/{archive_bytes}"
                or response.headers.get("Content-Encoding", "identity") != "identity"
                or int(response.headers.get("Content-Length", -1)) != length):
            raise PreparationError("provider_did_not_honor_exact_range")
        output = bytearray()
        for chunk in response.iter_content(CHUNK):
            if not isinstance(chunk, bytes) or len(output)+len(chunk) > length:
                raise PreparationError("range_body_overflow_or_nonbinary")
            output.extend(chunk)
        if len(output) != length:
            raise PreparationError("range_body_short")
        ledger.finish(verified_bytes=length)
        return bytes(output)
    except BaseException:
        ledger.finish()
        raise
    finally:
        if response is not None:
            response.close()


def download_component(item: dict, target: Path, *, session, ledger: Ledger) -> dict:
    """Bounded full-object direct GET, or exact ZIP ranges with streaming inflate."""
    partial = target.with_name(target.name + ".owned.partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    if partial.is_symlink() or target.exists():
        raise PreparationError("unsafe_or_existing_download_destination")
    partial.unlink(missing_ok=True)  # Caller has verified this worker's ownership marker.
    spec, expected = item["download"], item["bytes"]
    h, crc, total = hashlib.sha256(), 0, 0

    def write(stream, value):
        nonlocal crc, total
        if total+len(value) > expected:
            raise PreparationError("decoded_raw_exceeds_frozen_size")
        stream.write(value)
        h.update(value)
        crc = zlib.crc32(value, crc)
        total += len(value)

    with partial.open("xb") as stream:
        if spec["kind"] == "direct":
            ledger.reserve("provider_body_bytes", expected, item["path"])
            response = None
            try:
                response = session.get(spec["url"], headers={"Accept-Encoding": "identity"},
                                       stream=True, timeout=(20, 120))
                if (response.status_code != 200 or int(response.headers.get("Content-Length", -1)) != expected
                        or response.headers.get("Content-Encoding", "identity") != "identity"):
                    raise PreparationError("direct_get_status_or_declared_size_mismatch")
                for chunk in response.iter_content(CHUNK):
                    if not isinstance(chunk, bytes):
                        raise PreparationError("nonbinary_provider_body")
                    write(stream, chunk)
                if total != expected or h.hexdigest() != item["sha256"]:
                    raise PreparationError("direct_native_sha_or_size_mismatch")
                ledger.finish(verified_bytes=total)
            except BaseException:
                ledger.finish()
                raise
            finally:
                if response is not None:
                    response.close()
        elif spec["kind"] == "zip_range":
            header = _range(spec["url"], spec["header_offset"], 30, spec["archive_bytes"], session=session, ledger=ledger)
            signature, version, flags, method, mt, md, header_crc, csize, usize, nlen, xlen = struct.unpack("<IHHHHHIIIHH", header)
            if signature != 0x04034B50 or method != spec["compression"] or flags & 1 or method != 8:
                raise PreparationError("unsupported_or_changed_zip_local_header")
            tail = _range(spec["url"], spec["header_offset"]+30, nlen+xlen, spec["archive_bytes"], session=session, ledger=ledger)
            if tail[:nlen] != spec["member"].encode("utf-8"):
                raise PreparationError("zip_member_name_changed")
            if not flags & 8 and (header_crc != int(item["crc32"], 16) or csize != spec["compressed_bytes"] or usize != expected):
                raise PreparationError("zip_local_and_frozen_central_metadata_disagree")
            inflater = zlib.decompressobj(-15)
            offset = spec["header_offset"]+30+nlen+xlen
            remaining = spec["compressed_bytes"]
            while remaining:
                size = min(RANGE_CHUNK, remaining)
                compressed = _range(spec["url"], offset, size, spec["archive_bytes"], session=session, ledger=ledger)
                while compressed:
                    write(stream, inflater.decompress(compressed, CHUNK))
                    compressed = inflater.unconsumed_tail
                offset += size
                remaining -= size
            write(stream, inflater.flush(CHUNK))
            if not inflater.eof or inflater.unused_data or total != expected or f"{crc & 0xffffffff:08x}" != item["crc32"]:
                raise PreparationError("zip_eof_crc_or_decoded_size_mismatch")
        else:
            raise PreparationError("unapproved_download_kind")
        stream.flush()
        os.fsync(stream.fileno())
    if total != expected or h.hexdigest() != item["sha256"]:
        raise PreparationError("native_sha_or_size_mismatch")
    os.link(partial, target)
    _sync_parent(target)
    partial.unlink()
    return {"sha256": h.hexdigest(), "bytes": total, "crc32": f"{crc & 0xffffffff:08x}",
            "complete_original_body_or_exact_member_ranges": True}


def receipt_rows(root: Path, item: dict) -> list[dict]:
    _, restore = storage._modules()
    rows, _ = restore.load_verified_receipt(checked(root, item))
    return rows


def verify_raw_receipt(root: Path, item: dict, plan: dict, record: dict) -> list[dict]:
    rows = receipt_rows(root, item)
    if {(r["path"], r["sha256"], r["bytes"]) for r in rows} != {
            (r["path"], r["sha256"], r["bytes"]) for r in record["raw_files"]} or any(
            r["dataset"] != plan["dataset"] or r["role"] != "source_development_raw" for r in rows):
        raise PreparationError("raw_receipt_disagrees_with_frozen_selection")
    return rows


def verified_backup(root: Path, audit: Path, *, dataset: str, files: list[tuple[Path, str]],
                    prefix: str, ledger: Ledger, backup_fn) -> dict:
    manifest = {"schema_version": 1, "files": [{**descriptor(root, path), "dataset": dataset, "role": role}
                                               for path, role in files]}
    path, receipt = audit.with_name(audit.name+"_manifest.json"), audit.with_name(audit.name+"_receipt.json")
    atomic_json(path, manifest, immutable=True)
    reserve = sum(r["bytes"] for r in manifest["files"])
    ledger.reserve("r2_get_body_bytes", reserve, prefix)
    try:
        result = backup_fn(manifest_path=path, root=root, prefix=prefix, receipt_path=receipt, execute=True)
        if result.get("status") != "verified" or result.get("execute") is not True:
            raise PreparationError("r2_put_full_get_sha_not_verified")
        proof = descriptor(root, receipt)
        rows = receipt_rows(root, proof)
        fields = storage.ENTRY_FIELDS
        if {tuple(r[k] for k in fields) for r in rows} != {tuple(r[k] for k in fields) for r in manifest["files"]}:
            raise PreparationError("r2_receipt_object_set_changed")
        ledger.finish(verified_bytes=reserve)
        return proof
    except BaseException:
        ledger.finish()
        raise


def _state_verified(root: Path, state: dict, plan: dict, record: dict) -> bool:
    if state.get("status") != "COMPLETE":
        return False
    if state.get("plan_sha256") != plan["plan_sha256"] or state.get("record_id") != record["record_id"]:
        raise PreparationError("record_state_identity_changed")
    verify_raw_receipt(root, state["raw_backup_receipt"], plan, record)
    for item in state["artifacts"]:
        checked(root, item)
    rows = receipt_rows(root, state["cache_backup_receipt"])
    if {(r["path"], r["sha256"], r["bytes"]) for r in rows} != {
            (r["path"], r["sha256"], r["bytes"]) for r in state["artifacts"]}:
        raise PreparationError("cache_receipt_disagrees_with_local_artifacts")
    return True


def cleanup_owned(root: Path, state: dict, plan: dict, record: dict, audit: Path) -> None:
    if not _state_verified(root, state, plan, record):
        raise PreparationError("both_raw_and_cache_proofs_required_before_removal")
    for item in record["raw_files"]:
        if item["path"] not in state.get("owned_raw", []):
            continue
        marker = json.loads(local(root, (audit / (Path(item["path"]).name+".ownership.json")).relative_to(root).as_posix()).read_bytes())
        expected = {"created_by": CREATOR, "plan_sha256": plan["plan_sha256"], "record_id": record["record_id"], "path": item["path"]}
        if marker != expected:
            raise PreparationError("raw_ownership_unverified_preserve_file")
        raw = local(root, item["path"])
        if raw.exists():
            if sha_file(raw) != (item["sha256"], item["bytes"]):
                raise PreparationError("owned_raw_changed_preserve_file")
            raw.unlink()


def validate_native_cache(dataset: str, record: dict, native: dict, x, y, events: list, meta: dict) -> tuple[list, list, dict]:
    """Join all native trial identities to cache rows; validate actual unit facts.

    Yang's bound is explicitly a source analysis interval, with no claim that
    the physical MI onset or task end was independently measured.
    """
    raw_sha = record["raw_files"][0]["sha256"]
    if dataset == "Yang2025":
        gain = native.get("common_gain_evidence", {})
        if (gain.get("common_positive_gain_verified") is not True or gain.get("SI_unit_verified") is not False
                or native.get("derived_before_waveform_preprocessing") is not True
                or native.get("independent_second_parser_claim") is not False):
            raise PreparationError("yang_actual_common_gain_native_evidence_required")
        native_rows = native.get("events")
        unit = {"stored_unit": "calibrated_numeric_unknown_SI", "si_units_verified": False,
                "common_positive_gain_verified": True, "calibration": gain,
                "physical_MI_onset_verified": False, "physical_MI_end_verified": False}
        lookup = {r["event_uid"]: r for r in native_rows}
        compare = ("label_code", "original_event_index", "original_marker_sample",
                   "context_start_sample", "context_stop_sample_exclusive",
                   "crop_start_sample", "crop_stop_sample_exclusive")
        combined = events
    else:
        native_rows = native.get("trials")
        if (native.get("EEG_array_samples_read") != 0 or native.get("waveform_QC_applied") is not False
                or native.get("source_file_sha256") != raw_sha):
            raise PreparationError("dreyer_independent_original_tal_evidence_required")
        labels, units = native.get("native_signal_order", []), native.get("native_signal_units", {})
        if not isinstance(units, dict) or any(name not in labels or units.get(name) not in {"uV", "µV"} for name in CHANNELS):
            raise PreparationError("dreyer_actual_selected_native_units_unverified")
        unit = {"stored_unit": "V", "si_units_verified": True, "common_positive_gain_verified": True,
                "native_units": units, "native_signal_units": units, "selected_channels": CHANNELS,
                "SI_conversion": meta.get("SI_conversion"),
                "physical_MI_onset_verified": True, "physical_MI_end_verified": True}
        lookup = {r["trial_uid"]: r for r in native_rows}
        compare = ("label", "raw_label_code", "canonical_label", "original_trial_index",
                   "original_annotation_index", "analysis_anchor_sample", "task_end_sample",
                   "original_context_samples", "session_id", "physical_run_id", "source_file_sha256")
        combined = list(events) + list(meta.get("exclusions", []))
    if not isinstance(native_rows, list) or len(lookup) != len(native_rows):
        raise PreparationError("native_uid_missing_or_duplicate")
    normalized, cache_indices, seen = [], set(), set()
    for event in combined:
        event = dict(event)
        uid = event.get("event_uid", event.get("trial_uid"))
        if uid not in lookup or uid in seen:
            raise PreparationError("cache_native_event_membership_mismatch")
        seen.add(uid)
        original = lookup[uid]
        for field in compare:
            if field in original and event.get(field) != original[field]:
                raise PreparationError("cache_event_label_clock_or_raw_identity_mismatch")
        index = event.get("cache_row", event.get("cache_index"))
        valid = event.get("valid") is True if dataset == "Yang2025" else event.get("state") == "valid"
        event.update(event_uid=uid, valid=valid, cache_index=index, raw_sha256=raw_sha,
                     dataset=dataset, source_only=True, native_role=event.get("role"),
                     role="source_train", subject_id=record["subject_id"], session_id=record["session_id"],
                     input_contract_sha256=CONTRACT_SHA)
        if dataset == "Yang2025":
            event.update(original_trial_index=original["original_event_index"],
                raw_event_sha256=record["raw_files"][1]["sha256"],
                task_end_verified=False, MI_onset_verified=False,
                original_index_semantics="native_event_index_including_non_LR_events")
            if original.get("label") in (0, 1):
                event.update(canonical_label="left_hand" if original["label"] == 0 else "right_hand",
                    analysis_anchor_sample=original["context_start_sample"],
                    original_context_samples=[original["context_start_sample"], original["context_stop_sample_exclusive"]])
        else:
            event["task_end_verified"] = True
        if valid:
            if (type(index) is not int or not 0 <= index < len(x) or index in cache_indices
                    or event.get("label") != int(y[index])):
                raise PreparationError("cache_index_or_binary_label_mismatch")
            cache_indices.add(index)
            event["processed_signal_sha256"] = hashlib.sha256(x[index].tobytes(order="C")).hexdigest()
            qc = event.get("preprocessing", event.get("preprocessing_qc", {}))
            event["preprocessing_qc"] = dict(qc, stored_unit=unit["stored_unit"],
                si_units_verified=unit["si_units_verified"], common_positive_gain_verified=True,
                normalization_applied=False, reference="CAR_exact_clinic11")
        normalized.append(event)
    if seen != set(lookup) or cache_indices != set(range(len(x))):
        raise PreparationError("complete_native_event_and_valid_array_cover_required")
    if dataset == "Dreyer2023" and meta.get("independent_native_metadata_semantic_sha256") != digest(native):
        raise PreparationError("independent_native_metadata_semantic_sha_mismatch")
    return normalized, [e for e in normalized if not e["valid"]], unit


def process_record(root: Path, plan: dict, record: dict, *, native_profile: dict, ledger: Ledger,
                   session, materialize, audit_native=None, backup_fn=storage.run_backup, download_fn=download_component,
                   restore_fn=storage.run_restore) -> dict:
    dataset, plan_sha = plan["dataset"], plan["plan_sha256"]
    audit = local(root, f"audit/source_extensions_20261019/{dataset}/{plan_sha}/{record['record_id']}")
    output = local(root, f"{storage.RESULT_ROOT}/{dataset}/{plan_sha}/{record['record_id']}")
    audit.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    state_path = audit / "state.json"
    state = json.loads(state_path.read_bytes()) if state_path.exists() else {
        "status": "PENDING", "plan_sha256": plan_sha, "record_id": record["record_id"], "owned_raw": []}
    if state.get("plan_sha256") != plan_sha or state.get("record_id") != record["record_id"]:
        raise PreparationError("existing_record_state_identity_differs")
    for item in plan["code"].values():
        checked(root, item)
    if state.get("artifacts") and state.get("cache_backup_receipt"):
        candidate = dict(state, status="COMPLETE")
        if _state_verified(root, candidate, plan, record):
            cleanup_owned(root, candidate, plan, record, audit)
            atomic_json(state_path, candidate)
            return candidate
    try:
        needed = sum(r["bytes"] for r in record["raw_files"] if not local(root, r["path"]).exists())
        if shutil.disk_usage(root).free < needed+plan["budgets"]["disk_reserve_bytes"]:
            raise PreparationError("insufficient_disk_for_one_record_and_reserve")
        if state.get("raw_backup_receipt"):
            verify_raw_receipt(root, state["raw_backup_receipt"], plan, record)
            missing = [item for item in record["raw_files"] if not local(root, item["path"]).exists()]
            if missing:
                for item in missing:
                    binding = {"created_by": CREATOR, "plan_sha256": plan_sha,
                               "record_id": record["record_id"], "path": item["path"]}
                    atomic_json(audit/(Path(item["path"]).name+".ownership.json"), binding, immutable=True)
                    if item["path"] not in state["owned_raw"]:
                        state["owned_raw"].append(item["path"])
                atomic_json(state_path, state)
                reserve = sum(item["bytes"] for item in missing)
                ledger.reserve("r2_get_body_bytes", reserve, record["record_id"]+":raw_restore")
                try:
                    result = restore_fn(backup_receipt_path=checked(root, state["raw_backup_receipt"]),
                        root=root, report_path=audit/"raw_restore_report.json", paths=[item["path"] for item in missing],
                        execute=True, require_fresh=True)
                    if result.get("status") != "restored_verified" or result.get("fresh_full_get_verified") is not True:
                        raise PreparationError("raw_r2_restore_full_get_not_verified")
                    ledger.finish(verified_bytes=reserve)
                except BaseException:
                    ledger.finish()
                    raise
        for item in record["raw_files"]:
            raw = local(root, item["path"])
            marker_path = audit / (raw.name+".ownership.json")
            binding = {"created_by": CREATOR, "plan_sha256": plan_sha, "record_id": record["record_id"], "path": item["path"]}
            if raw.exists():
                try:
                    verify_raw_file(raw, item)
                except PreparationError:
                    raise PreparationError("existing_raw_different_preserved")
                if marker_path.exists():
                    if json.loads(marker_path.read_bytes()) != binding:
                        raise PreparationError("existing_ownership_marker_mismatch")
                    if item["path"] not in state["owned_raw"]:
                        state["owned_raw"].append(item["path"])
                # No marker means a pre-existing pilot: never adopt ownership.
                elif item["path"] not in {r["path"] for r in state.get("reused_existing_raw", [])}:
                    state.setdefault("reused_existing_raw", []).append({
                        "path": item["path"], "sha256": item["sha256"], "bytes": item["bytes"],
                        "actual_local_full_sha256_rechecked": True, "new_provider_GET_performed": False,
                        "ownership_adopted": False, "historical_get_evidence": plan.get("historical_get_evidence", [])})
                    atomic_json(state_path, state)
            else:
                if raw.with_name(raw.name+".owned.partial").exists() and not marker_path.exists():
                    raise PreparationError("unowned_existing_partial_preserved")
                atomic_json(marker_path, binding, immutable=True)
                if item["path"] not in state["owned_raw"]:
                    state["owned_raw"].append(item["path"])
                atomic_json(state_path, state)
                info = download_fn(item, raw, session=session, ledger=ledger)
                if info["sha256"] != item["sha256"] or info["bytes"] != item["bytes"]:
                    raise PreparationError("download_proof_disagrees_with_frozen_native_identity")
            verify_raw_file(raw, item)
        if not state.get("raw_backup_receipt"):
            state["raw_backup_receipt"] = verified_backup(root, audit / "raw_backup", dataset=dataset,
                files=[(local(root, item["path"]), "source_development_raw") for item in record["raw_files"]],
                prefix=f"paper3/source-extensions-20261019/{dataset}/{plan_sha}/{record['record_id']}/raw/",
                ledger=ledger, backup_fn=backup_fn)
            atomic_json(state_path, state)
        verify_raw_receipt(root, state["raw_backup_receipt"], plan, record)
        raw_paths = [local(root, item["path"]) for item in record["raw_files"]]
        expected_sha = ({Path(item["path"]).name: item["sha256"] for item in record["raw_files"]}
                        if dataset == "Yang2025" else record["raw_files"][0]["sha256"])
        native = (audit_native(*raw_paths, expected_sha256=expected_sha, profile=native_profile)
                  if audit_native is not None else None)
        x, y, events, meta = materialize(*raw_paths, expected_sha256=expected_sha, profile=native_profile)
        if dataset == "Yang2025":
            native = meta.pop("native_metadata", None)
        x, y = np.asarray(x), np.asarray(y)
        for item in plan["code"].values():
            checked(root, item)
        if (x.dtype != np.float32 or x.ndim != 3 or x.shape[1:] != (11, 128) or not np.isfinite(x).all()
                or y.dtype != np.int64 or y.shape != (len(x),) or not np.isin(y, [0, 1]).all()):
            raise PreparationError("native_cache_array_contract_failed")
        if not isinstance(events, list) or not isinstance(meta, dict) or not isinstance(native, dict):
            raise PreparationError("native_metadata_interface_invalid")
        uids = [e.get("event_uid", e.get("trial_uid")) for e in events]
        if not all(isinstance(uid, str) and uid for uid in uids) or len(uids) != len(set(uids)):
            raise PreparationError("native_event_uid_missing_or_duplicate")
        events, exclusions, unit = validate_native_cache(dataset, record, native, x, y, events, meta)
        meta = dict(meta, dataset=dataset, subject_id=record["subject_id"], session_id=record["session_id"],
            role="source_train", source_only=True, stage=STAGE, channels=CHANNELS,
            input_contract_sha256=CONTRACT_SHA, normalization_applied=False, synthetic=False,
            old_npz_reused=False, raw_backup_receipt=state["raw_backup_receipt"],
            raw_files=record["raw_files"], adapter_code=plan["code"][ADAPTERS[dataset].replace(".", "/")+".py"])
        paths = {name: output/(name+".npy" if name in {"X", "y"} else name+".json")
                 for name in ("X", "y", "events", "exclusions", "metadata")}
        paths.update(native_metadata=audit/"native_metadata.json", unit_evidence=audit/"unit_evidence.json")
        atomic_json(paths["native_metadata"], native, immutable=True)
        unit.update(schema_version="paper3-source-native-unit-evidence-v1", dataset=dataset,
                    native_metadata=descriptor(root, paths["native_metadata"]), native_profile=plan["native_profile"])
        atomic_json(paths["unit_evidence"], unit, immutable=True)
        unit_desc = descriptor(root, paths["unit_evidence"])
        for event in events:
            event["unit_evidence"] = unit_desc
        meta.update(native_metadata=descriptor(root, paths["native_metadata"]), unit_evidence=unit_desc,
                    stored_unit=unit["stored_unit"], si_units_verified=unit["si_units_verified"],
                    common_positive_gain_verified=True)
        array_once(paths["X"], x)
        array_once(paths["y"], y)
        for name, value in (("events", events), ("exclusions", exclusions), ("metadata", meta),
                            ("native_metadata", native), ("unit_evidence", unit)):
            atomic_json(paths[name], value, immutable=True)
        artifacts = [descriptor(root, path) for path in paths.values()]
        state.update(status="CACHE_LOCAL_VERIFIED", artifacts=artifacts)
        atomic_json(state_path, state)
        receipt = verified_backup(root, audit / "cache_backup", dataset=dataset,
            files=[(path, "result" if name in {"X", "y", "events", "exclusions", "metadata"} else "audit")
                   for name, path in paths.items()],
            prefix=f"paper3/source-extensions-20261019/{dataset}/{plan_sha}/{record['record_id']}/cache/",
            ledger=ledger, backup_fn=backup_fn)
        shard = {"subject_id": record["subject_id"], "session_id": record["session_id"],
            "record_id": record["record_id"], "valid_trials": len(x), "original_events": len(events),
            **{name: descriptor(root, path) for name, path in paths.items()},
            "adapter_code": meta["adapter_code"], "raw_files": record["raw_files"],
            "raw_sha256": record["raw_files"][0]["sha256"],
            "native_profile": plan["native_profile"],
            "processing_code": plan["code"].get("p3work/preprocess.py"),
            "raw_backup_receipt": state["raw_backup_receipt"], "cache_backup_receipt": receipt}
        state.update(status="COMPLETE", cache_backup_receipt=receipt, shard=shard)
        atomic_json(state_path, state)
        cleanup_owned(root, state, plan, record, audit)
        state["owned_raw_removed_after_dual_full_get_proofs"] = True
        atomic_json(state_path, state)
        return state
    except BaseException as exc:
        ledger.finish()
        state.update(status="FAILED_CLOSED", error_type=type(exc).__name__,
            error_code=str(exc) if isinstance(exc, PreparationError) else "operation_failed_details_suppressed")
        atomic_json(state_path, state)
        raise PreparationError(state["error_code"]) from None


def assemble_manifest(root: Path, plan: dict, states: list[dict]) -> dict:
    if len(states) != plan["record_count"] or {s["record_id"] for s in states} != {r["record_id"] for r in plan["records"]}:
        raise PreparationError("all_frozen_records_required_before_complete")
    lookup = {r["record_id"]: r for r in plan["records"]}
    seen, shards = set(), []
    for state in states:
        if not _state_verified(root, state, plan, lookup[state["record_id"]]):
            raise PreparationError("unverified_record_cannot_enter_manifest")
        shard = state["shard"]
        for event in json.loads(checked(root, shard["events"]).read_bytes()):
            uid = event.get("event_uid", event.get("trial_uid"))
            if uid in seen:
                raise PreparationError("cross_record_event_uid_duplicate")
            seen.add(uid)
        shards.append(shard)
    return {"schema_version": SCHEMA, "status": "SOURCE_EXTENSION_CACHE_COMPLETE", "dataset": plan["dataset"],
        "task_subset": plan["task_subset"], "role": "source_train", "unique_role_within_branch": True,
        "source_only": True, "channels": CHANNELS, "stage": STAGE, "input_contract_sha256": CONTRACT_SHA,
        "synthetic": False, "old_npz_reused": False, "normalization_applied": False, "training_started": False,
        "external_payload_accessed": False, "plan_sha256": plan["plan_sha256"], "native_profile": plan["native_profile"],
        "processing_code": plan["code"].get("p3work/preprocess.py"),
        "coordinator_code": plan["code"].get("p3work/source_extension_coordinator.py"),
        "official_inventory": plan["inventory"], "record_count": len(shards),
        "provider_subject_ids": len({s["subject_id"] for s in shards}),
        "biological_independent_people_verified": False,
        "valid_trials": sum(s["valid_trials"] for s in shards), "shards": shards}


def prepare(root: Path, plan: dict, *, execute=False, max_new_records=None,
            session=None, materialize=None, audit_native=None, backup_fn=storage.run_backup,
            download_fn=download_component) -> dict:
    root = Path(os.path.abspath(root))
    native_profile = validate_plan(root, plan)
    if max_new_records is not None and (type(max_new_records) is not int or max_new_records < 1):
        raise PreparationError("positive_record_limit_required")
    if not execute:
        return {"status": "DRY_RUN_FROZEN_SOURCE_PLAN_NO_NETWORK", "plan_sha256": plan["plan_sha256"],
            "dataset": plan["dataset"], "records": plan["record_count"], "budgets": plan["budgets"],
            "training_started": False, "GPU_actions": 0, "external_payload_accessed": False}
    coordinator_root = local(root, "audit/source_extensions_20261019")
    coordinator_root.mkdir(parents=True, exist_ok=True)
    lock_path = coordinator_root / ".source-extension-singleton.lock"
    local(root, lock_path.relative_to(root).as_posix())
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PreparationError("source_extension_coordinator_already_running") from None
        dataset, sha = plan["dataset"], plan["plan_sha256"]
        audit = local(root, f"audit/source_extensions_20261019/{dataset}/{sha}")
        output = local(root, f"{storage.RESULT_ROOT}/{dataset}/{sha}")
        audit.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(audit/"frozen_plan.json", plan, immutable=True)
        ledger = Ledger(audit/"budget_ledger.json", plan)
        if materialize is None or (audit_native is None and dataset == "Dreyer2023"):
            adapter = importlib.import_module(ADAPTERS[dataset])
            materialize = materialize or adapter.materialize
            audit_native = audit_native or getattr(adapter, "audit_native", None)
        if session is None:
            import requests
            session = requests.Session()
        states, new = [], 0
        try:
            for record in plan["records"]:
                ledger.checkpoint()
                state_path = audit / record["record_id"] / "state.json"
                previous = json.loads(state_path.read_bytes()) if state_path.exists() else {}
                complete = previous.get("status") == "COMPLETE"
                if not complete and max_new_records is not None and new >= max_new_records:
                    break
                state = process_record(root, plan, record, native_profile=native_profile, ledger=ledger,
                    session=session, materialize=materialize, audit_native=audit_native,
                    backup_fn=backup_fn, download_fn=download_fn)
                states.append(state)
                new += int(not complete)
                cache_bytes = sum(r["bytes"] for s in states for r in s["artifacts"])
                if cache_bytes > plan["budgets"]["cache_bytes"]:
                    raise PreparationError("cumulative_cache_disk_budget_exhausted")
            if len(states) != plan["record_count"]:
                result = {"status": "PAUSED_EXPLICIT_RECORD_LIMIT", "verified_records": len(states),
                    "total_records": plan["record_count"], "plan_sha256": sha, "training_started": False}
            else:
                manifest = assemble_manifest(root, plan, states)
                native_inventory = {"schema_version": "paper3-source-extension-native-inventory-v1", "dataset": dataset,
                    "records": [{"record_id": s["record_id"], "subject_id": s["subject_id"], "session_id": s["session_id"],
                        "raw_files": s["raw_files"], "native_metadata": s["native_metadata"],
                        "raw_sha256": s["raw_sha256"], "raw_bytes": s["raw_files"][0]["bytes"],
                        "raw_backup_receipt": s["raw_backup_receipt"]} for s in manifest["shards"]]}
                atomic_json(audit/"native_inventory.json", native_inventory, immutable=True)
                unit_inventory = {"dataset": dataset, "records": [s["unit_evidence"] for s in manifest["shards"]]}
                atomic_json(audit/"unit_evidence.json", unit_inventory, immutable=True)
                manifest.update(native_inventory=descriptor(root, audit/"native_inventory.json"),
                                unit_evidence=descriptor(root, audit/"unit_evidence.json"))
                assignment = {"dataset": dataset, "role": "source_train", "source_only": True,
                    "unique_role_within_branch": True, "development": False, "longitudinal_meta_ready": False,
                    "external_evaluation": False, "new_unseen_external_claim": False,
                    "document_sha256": storage.DOCUMENT_SHA256, "plan_sha256": sha}
                atomic_json(audit/"role_assignment.json", assignment, immutable=True)
                manifest["role_assignment"] = descriptor(root, audit/"role_assignment.json")
                manifest_path = output/"source_cache.json"
                atomic_json(manifest_path, manifest, immutable=True)
                receipt = verified_backup(root, audit/"final_backup", dataset=dataset,
                    files=[(manifest_path, "result"), (audit/"native_inventory.json", "audit"),
                           (audit/"unit_evidence.json", "audit"), (audit/"frozen_plan.json", "audit"),
                           (audit/"role_assignment.json", "audit")],
                    prefix=f"paper3/source-extensions-20261019/{dataset}/{sha}/final/", ledger=ledger, backup_fn=backup_fn)
                result = {"status": "SOURCE_EXTENSION_CACHE_COMPLETE", "dataset": dataset, "manifest": descriptor(root, manifest_path),
                    "backup_receipt": receipt, "record_count": len(states), "valid_trials": manifest["valid_trials"],
                    "training_started": False, "GPU_actions": 0, "external_payload_accessed": False}
            atomic_json(audit/"coordinator_status.json", result)
            return result
        finally:
            ledger.checkpoint(close=True)
