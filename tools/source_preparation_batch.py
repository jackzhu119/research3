#!/usr/bin/env python3
"""Serial, resumable official source raw/metadata preparation; default dry run.

The 505 URLs are candidates, never a claim of existence. Execution is W1-gated.
Every observed object needs an actual HEAD, a complete GET/local SHA256, native
schema audit, and explicit-manifest R2 PUT plus complete GET/SHA256 evidence.
This worker does not create a numerical cache, train, or contact RunPod. Only
new tool-owned temporary raw may be removed after verified R2 archival; the
four existing source_schema specimens and all unowned files are preserved.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import urllib.request

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from p3work.adapters.source_native import audit_native_source
from p3work.contracts import digest, file_digest, validate_signal_contract
from p3work.recovery_budget import PersistentWallBudget, WallBudgetError
from tools.prepare_source_cache import expected_source_objects
from tools.r2_backup import create_client, run_backup, sha256_file, utc_now
from tools.r2_restore import load_verified_receipt, no_symlink_components

CHUNK = 4 * 1024**2
PREFIX = "paper3/source-preparation-v2/"
SOURCE_ADAPTER_PATH = Path(sys.modules[audit_native_source.__module__].__file__).resolve()
_NATIVE_BACKUP_IMPLEMENTATION = run_backup
VERSIONS = {
    "BNCI2014_001": "001-2014, unversioned native MAT",
    "Cho2017": "GigaDB DOI snapshot retrieved in receipt",
    "Lee2019_MI": "GigaDB DOI snapshot retrieved in receipt",
    "PhysionetMI": "1.0.0",
}


class BatchError(Exception):
    """Only fixed non-sensitive codes may be emitted to logs or state."""


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".source-batch-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def descriptor(root: Path, path: Path) -> dict:
    no_symlink_components(path)
    return {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}


def local_path(root: Path, relative: str) -> Path:
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or "\\" in relative or any(p in {"", ".", ".."} for p in relative.split("/"))):
        raise BatchError("unsafe_root_relative_path")
    path = root / relative
    no_symlink_components(path)
    if not path.resolve().is_relative_to(root):
        raise BatchError("path_outside_independent_paper3_root")
    return path


def build_plan(*, max_download_bytes=100 * 1024**3, max_single_raw_bytes=2 * 1024**3,
               max_wall_seconds=86400, min_free_disk_bytes=512 * 1024**2,
               min_available_ram_bytes=1024**3, parser_ram_multiplier=6) -> dict:
    """Pure metadata; no local EEG, network, credentials, or stage mutations."""
    limits = dict(max_download_bytes=max_download_bytes, max_single_raw_bytes=max_single_raw_bytes,
                  max_wall_seconds=max_wall_seconds, min_free_disk_bytes=min_free_disk_bytes,
                  min_available_ram_bytes=min_available_ram_bytes, parser_ram_multiplier=parser_ram_multiplier)
    if any(type(v) is not int or v <= 0 for v in limits.values()):
        raise BatchError("resource_budgets_must_be_positive_integers")
    files = []
    for identity, url in sorted(expected_source_objects().items()):
        dataset, subject, session, name = identity
        files.append({"object_id": digest(list(identity)), "dataset": dataset,
                      "provider_subject": subject, "session_id": session, "file": name,
                      "version": VERSIONS[dataset], "url": url,
                      "existence_verified": False, "bytes": None, "sha256": None})
    if len(files) != 505:
        raise BatchError("exact_505_source_candidates_required")
    plan = {"schema_version": "paper3-source-preparation-plan-v1", "status": "CANDIDATE_PLAN_ONLY",
            "adapter_code_sha256": file_digest(SOURCE_ADAPTER_PATH),
            "files": files, "candidate_count": len(files), "provider_roster_verified": False,
            "candidate_sizes_known": False, "inflight_downloads": 1, "parallel_native_parsers": 1,
            "download_byte_budget_scope": "official-provider full-GET reservations including failed/unclean attempts; observed bytes reported separately; independent R2 caps",
            "max_jobs": len(files), "max_gpu_hours": 0, **limits,
            "raw_staging_relative_root": "data/source_preparation/staging",
            "raw_removal": "only tool-owned temporary raw after full R2 PUT/GET/SHA256 verification",
            "existing_source_schema_specimens": "preserve permanently",
            "numeric_cache_created": False, "training_started": False,
            "next_stage": "prepare_source_cache.py; independently gated numerical cache, no automatic training"}
    plan["plan_sha256"] = digest(plan)
    return plan


def validate_plan(plan: dict) -> None:
    names = ("max_download_bytes", "max_single_raw_bytes", "max_wall_seconds",
             "min_free_disk_bytes", "min_available_ram_bytes", "parser_ram_multiplier")
    try:
        expected = build_plan(**{k: plan[k] for k in names})
    except (KeyError, TypeError):
        raise BatchError("invalid_candidate_plan") from None
    if plan != expected:
        raise BatchError("candidate_plan_not_exact_registered_505_objects")


def execution_gate(root: Path, contract_path="contracts/input_contract.json") -> str:
    tasks = json.loads((root / "status/tasks.json").read_text())
    if tasks.get("W1", {}).get("status") != "COMPLETE":
        raise BatchError("w1_complete_required_before_source_batch_execution")
    try:
        return validate_signal_contract(json.loads(local_path(root, contract_path).read_text()))
    except ValueError:
        raise BatchError("frozen_valid_signal_contract_required") from None


class ResourceBudget:
    """Durable wall plus separately capped full-body reservation ledgers.

    Observed bytes are only bytes whose post-read record was persisted. Charged
    bytes additionally include *all* of an unclosed/failed GET reservation, so a
    SIGKILL between read() and accounting cannot refund that transfer. The
    official-provider cap does not cover R2: backup verification and cache raw
    restoration have independent scopes and explicit limits.
    """

    OFFICIAL = "official_provider_GET_body"
    R2_BACKUP = "R2_backup_full_GET_body"
    R2_RESTORE = "R2_cache_raw_restore_GET_body"

    def __init__(self, plan, progress_path, *, clock=time.monotonic, utc_clock=None, boot_id=None,
                 scope=OFFICIAL, identity_sha256=None, max_download_bytes=None,
                 max_wall_seconds=None, max_r2_readback_bytes=None, amendment_note=None):
        self.plan, self.path, self.clock, self.scope = plan, Path(progress_path), clock, scope
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = self.path.with_name(self.path.name + ".lock").open("a")
        self._closed = False
        try:
            try:
                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise BatchError("source_resource_budget_already_active") from None
            prior = json.loads(self.path.read_text()) if self.path.exists() else None
            if prior is not None and not isinstance(prior, dict):
                raise BatchError("invalid_persisted_resource_journal")
            prior_scopes = prior.get("byte_scopes", {}) if prior else {}
            maximum = prior_scopes.get(scope, {}).get("limit_bytes", plan["max_download_bytes"]) if max_download_bytes is None else max_download_bytes
            wall_file = self.path.with_name(self.path.stem + ".wall.json")
            saved_wall = json.loads(wall_file.read_text()) if wall_file.exists() else {}
            wall_maximum = saved_wall.get("max_wall_seconds", plan["max_wall_seconds"]) if max_wall_seconds is None else max_wall_seconds
            limits = {scope: maximum}
            if scope == self.OFFICIAL:
                limits[self.R2_BACKUP] = prior_scopes.get(self.R2_BACKUP, {}).get("limit_bytes", 2 * plan["max_download_bytes"]) if max_r2_readback_bytes is None else max_r2_readback_bytes
            if any(type(v) is not int or v <= 0 for v in limits.values()):
                raise BatchError("explicit_positive_byte_limits_required")
            resource_identity = identity_sha256 or digest({"schema": "source-resource-identity-v2",
                "adapter_code_sha256": plan.get("adapter_code_sha256"), "files": plan.get("files"), "scope": scope})
            if prior is not None:
                if (prior.get("schema_version") != "paper3-source-resource-budget-v2" or
                        prior.get("resource_identity_sha256") != resource_identity or prior.get("primary_scope") != scope or
                        set(prior.get("byte_scopes", {})) != set(limits)):
                    raise BatchError("resource_identity_or_legacy_untracked_intent_mismatch_preserved")
                self.data = prior
                for name, ledger in prior["byte_scopes"].items():
                    for key in ("observed_bytes", "charged_bytes", "limit_bytes", "initial_limit_bytes", "next_sequence"):
                        if type(ledger.get(key)) is not int or ledger[key] < (1 if key in {"limit_bytes", "initial_limit_bytes", "next_sequence"} else 0):
                            raise BatchError("invalid_persisted_byte_ledger")
                    if ledger["observed_bytes"] > ledger["charged_bytes"] or not isinstance(ledger.get("attempts"), list):
                        raise BatchError("invalid_persisted_byte_high_water")
                    sequences = set(); observed_sum = 0; charged_sum = 0
                    attempts = ledger["attempts"] + ([ledger["active"]] if ledger.get("active") is not None else [])
                    for attempt in attempts:
                        if (not isinstance(attempt, dict) or type(attempt.get("sequence")) is not int or
                                not 0 < attempt["sequence"] < ledger["next_sequence"] or attempt["sequence"] in sequences or
                                attempt.get("scope") != name or type(attempt.get("reserved_bytes")) is not int or attempt["reserved_bytes"] <= 0 or
                                type(attempt.get("expected_body_bytes")) is not int or not 0 < attempt["expected_body_bytes"] <= attempt["reserved_bytes"] or
                                type(attempt.get("observed_bytes")) is not int or not 0 <= attempt["observed_bytes"] <= attempt["reserved_bytes"]):
                            raise BatchError("invalid_persisted_GET_reservation")
                        sequences.add(attempt["sequence"]); observed_sum += attempt["observed_bytes"]
                        charged_sum += attempt["observed_bytes"] if attempt.get("status") == "COMPLETE_OBSERVED_BODY" else attempt["reserved_bytes"]
                    if (observed_sum, charged_sum) != (ledger["observed_bytes"], ledger["charged_bytes"]):
                        raise BatchError("persisted_byte_counter_differs_from_durable_attempts")
                    limit = ledger["initial_limit_bytes"]
                    if not isinstance(ledger.get("amendments"), list):
                        raise BatchError("invalid_persisted_byte_amendments")
                    for amendment in ledger["amendments"]:
                        if (amendment.get("previous_limit_bytes") != limit or type(amendment.get("new_limit_bytes")) is not int or
                                amendment["new_limit_bytes"] <= limit or amendment.get("scientific_identity_changed") is not False or
                                not isinstance(amendment.get("note"), str) or not amendment["note"].strip()):
                            raise BatchError("invalid_persisted_byte_amendments")
                        limit = amendment["new_limit_bytes"]
                    if limit != ledger["limit_bytes"]:
                        raise BatchError("byte_limit_differs_from_explicit_amendments")
                    if ledger.get("active") is not None:
                        attempt = ledger["active"]
                        if (type(attempt.get("reserved_bytes")) is not int or attempt["reserved_bytes"] <= 0 or
                                type(attempt.get("observed_bytes")) is not int or not 0 <= attempt["observed_bytes"] <= attempt["reserved_bytes"]):
                            raise BatchError("invalid_persisted_GET_reservation")
                        ledger["attempts"].append({**attempt, "status": "UNCLEAN_FULL_RESERVATION_REMAINS_CHARGED"})
                        ledger["active"] = None
                    self._amend_byte_limit(name, limits[name], amendment_note)
            else:
                self.data = {"schema_version": "paper3-source-resource-budget-v2", "resource_identity_sha256": resource_identity,
                    "primary_scope": scope, "plan_sha256": plan["plan_sha256"], "byte_scopes": {},
                    "active_wall_elapsed_s": 0.0, "max_gpu_hours": 0,
                    "accounting": "observed_post_read_bytes_separate_from_conservative_full_GET_reservations",
                    "scope_definitions": {self.OFFICIAL: "provider GET body plus at most one explicit EOF probe byte per reservation",
                        self.R2_BACKUP: "backup verification full GET body; default declared cap=2*initial provider cap, not a measured source-size estimate",
                        self.R2_RESTORE: "cache native-raw GET body only; independent inventory-derived limit; excludes provider GET and backup verification"},
                    "limits_are_operational_only": True}
                for name, maximum in limits.items():
                    self.data["byte_scopes"][name] = {"initial_limit_bytes": maximum, "limit_bytes": maximum,
                        "observed_bytes": 0, "charged_bytes": 0, "active": None, "next_sequence": 1,
                        "attempts": [], "amendments": []}
            options = {"monotonic": clock, "boot_id": boot_id, "amendment_note": amendment_note}
            if utc_clock is not None:
                options["utc_now"] = utc_clock
            self.wall = PersistentWallBudget(wall_file, resource_identity,
                wall_maximum, checkpoint_wall_seconds=self.data["active_wall_elapsed_s"], **options)
            if not self.wall.blocked and not self.wall.exhausted:
                self.wall.start_interval()
            self._persist()
        except BaseException as error:
            if hasattr(self, "wall"):
                self.wall.close()
            self._lock.close(); self._closed = True
            if isinstance(error, WallBudgetError):
                raise BatchError("source_wall_budget_identity_clock_or_explicit_amendment_required") from None
            raise

    def _amend_byte_limit(self, scope, maximum, note):
        ledger = self.data["byte_scopes"][scope]
        if maximum == ledger["limit_bytes"]:
            return
        if maximum <= ledger["limit_bytes"] or not isinstance(note, str) or not note.strip():
            raise BatchError("byte_limit_change_requires_explicit_larger_operational_amendment")
        ledger["amendments"].append({"previous_limit_bytes": ledger["limit_bytes"], "new_limit_bytes": maximum,
            "note": note.strip(), "scientific_identity_changed": False})
        ledger["limit_bytes"] = maximum

    def _persist(self):
        ledger = self.data["byte_scopes"][self.scope]
        self.data.update(download_bytes_read=ledger["observed_bytes"], download_bytes_charged=ledger["charged_bytes"])
        if hasattr(self, "wall"):
            report = self.wall.report()
            self.data["active_wall_elapsed_s"] = max(self.data["active_wall_elapsed_s"], report["wall_seconds"])
            self.data["wall_budget"] = report
        atomic_json(self.path, self.data)

    def checkpoint(self):
        try:
            self.wall.checkpoint(self.data["active_wall_elapsed_s"])
        except WallBudgetError:
            self._persist()
            raise BatchError("source_wall_clock_unknown_no_stage_advance") from None
        self._persist()

    def check(self, *, required_bytes=0, scope=None):
        self.checkpoint()
        if self.wall.blocked:
            raise BatchError("source_wall_clock_unknown_no_stage_advance")
        if self.wall.exhausted:
            raise BatchError("wall_budget_exhausted")
        ledger = self.data["byte_scopes"][scope or self.scope]
        if type(required_bytes) is not int or required_bytes < 0:
            raise BatchError("invalid_required_byte_reservation")
        if ledger["charged_bytes"] + required_bytes > ledger["limit_bytes"]:
            raise BatchError("download_byte_budget_exhausted")

    def reserve_transfer(self, expected_bytes, *, object_id, scope=None, probe_bytes=0):
        scope = scope or self.scope
        if (type(expected_bytes) is not int or expected_bytes <= 0 or type(probe_bytes) is not int or probe_bytes < 0 or
                not isinstance(object_id, str) or not object_id):
            raise BatchError("complete_known_positive_GET_body_reservation_required")
        reserved = expected_bytes + probe_bytes
        self.check(required_bytes=reserved, scope=scope)
        if any(row["active"] is not None for row in self.data["byte_scopes"].values()):
            raise BatchError("serial_GET_reservation_already_open")
        ledger = self.data["byte_scopes"][scope]
        attempt = {"sequence": ledger["next_sequence"], "object_id": object_id,
            "expected_body_bytes": expected_bytes, "reserved_bytes": reserved, "observed_bytes": 0,
            "scope": scope, "intent": "durably_reserved_before_GET_or_body_read"}
        ledger["next_sequence"] += 1; ledger["charged_bytes"] += reserved; ledger["active"] = attempt
        self._persist()
        return attempt["sequence"]

    def consume(self, count, *, transfer_id=None, scope=None):
        scope = scope or self.scope
        ledger = self.data["byte_scopes"][scope]; active = ledger["active"]
        if (active is None or transfer_id != active["sequence"] or type(count) is not int or count < 0):
            raise BatchError("body_read_requires_matching_durable_GET_reservation")
        active["observed_bytes"] += count; ledger["observed_bytes"] += count
        excess = max(0, active["observed_bytes"] - active["reserved_bytes"])
        if excess:
            ledger["charged_bytes"] += excess; active["reserved_bytes"] += excess
        self._persist()  # Persist observed traffic before checking a wall/cap stop.
        self.check(scope=scope)

    def finish_transfer(self, transfer_id, *, complete_body_read=False, scope=None):
        scope = scope or self.scope
        ledger = self.data["byte_scopes"][scope]; active = ledger["active"]
        if active is None or transfer_id != active["sequence"]:
            raise BatchError("GET_completion_without_matching_reservation")
        exact = complete_body_read is True and active["observed_bytes"] == active["expected_body_bytes"]
        if exact:
            ledger["charged_bytes"] -= active["reserved_bytes"] - active["observed_bytes"]
        ledger["attempts"].append({**active, "status": "COMPLETE_OBSERVED_BODY" if exact else "FAILED_OR_UNKNOWN_FULL_RESERVATION_REMAINS_CHARGED"})
        ledger["active"] = None
        self._persist()

    def timeout(self):
        self.check()
        return max(0.1, min(90, self.wall.remaining_seconds))

    def close(self):
        if not self._closed:
            try:
                self.wall.graceful_close(self.data["active_wall_elapsed_s"])
                self._persist()
            finally:
                self.wall.close(); self._lock.close(); self._closed = True

    def __del__(self):
        if hasattr(self, "_lock") and not self._lock.closed:
            self._lock.close()  # No graceful-close claim when a handle is discarded.


class _BudgetedR2Body:
    def __init__(self, body, budget, transfer_id, scope, expected_bytes):
        self.body, self.budget, self.transfer_id, self.scope = body, budget, transfer_id, scope
        self.remaining = expected_bytes
        self.eof_probed = False

    def read(self, amount=-1):
        self.budget.check(scope=self.scope)
        if self.remaining == 0:
            if self.eof_probed:
                return b""
            block = self.body.read(1)  # Exactly one additionally reserved byte, never an unbounded EOF read.
            if not isinstance(block, bytes):
                raise BatchError("non_binary_R2_GET_EOF_probe")
            self.budget.consume(len(block), transfer_id=self.transfer_id, scope=self.scope)
            self.eof_probed = True
            if block:
                raise BatchError("R2_GET_body_exceeds_exact_reserved_ContentLength")
            return b""
        count = self.remaining if amount is None or amount < 0 else min(amount, self.remaining)
        block = self.body.read(count)
        if not isinstance(block, bytes):
            raise BatchError("non_binary_R2_GET_body")
        self.budget.consume(len(block), transfer_id=self.transfer_id, scope=self.scope)
        if len(block) > count:
            raise BatchError("R2_stream_violated_bounded_read_contract")
        self.remaining -= len(block)
        return block

    def __getattr__(self, name):
        return getattr(self.body, name)


class _WallCheckedUpload:
    def __init__(self, body, budget, scope):
        self.body, self.budget, self.scope = body, budget, scope

    def read(self, amount=-1):
        self.budget.check(scope=self.scope)
        return self.body.read(amount)

    def __getattr__(self, name):
        return getattr(self.body, name)


class BudgetedR2Client:
    """Configured SDK proxy; no global helper or credential changes.

    Require exact expected response length before reading, then bound each read
    to its remaining framed body and persist observed bytes/wall checks. PUT
    body reads check wall limits but remain outside GET byte counters.
    """
    def __init__(self, client, budget, transfer_id, expected_gets, *, scope):
        self.client, self.budget, self.transfer_id = client, budget, transfer_id
        self.expected_gets, self.scope = dict(expected_gets), scope

    def get_object(self, **kwargs):
        key = kwargs.get("Key")
        if key not in self.expected_gets:
            raise BatchError("R2_GET_not_in_exact_reserved_manifest")
        self.budget.check(scope=self.scope)
        response = self.client.get_object(**kwargs)
        body, expected = response.get("Body"), self.expected_gets[key]
        if (response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 200 or
                type(response.get("ContentLength")) is not int or response["ContentLength"] != expected or body is None):
            if body is not None:
                body.close()
            raise BatchError("R2_GET_requires_200_and_exact_reserved_ContentLength_before_read")
        return {**response, "Body": _BudgetedR2Body(body, self.budget, self.transfer_id, self.scope, expected)}

    def put_object(self, **kwargs):
        self.budget.check(scope=self.scope)
        if hasattr(kwargs.get("Body"), "read"):
            kwargs["Body"] = _WallCheckedUpload(kwargs["Body"], self.budget, self.scope)
        return self.client.put_object(**kwargs)

    def __getattr__(self, name):
        return getattr(self.client, name)


def check_space(root: Path, size: int, plan: dict) -> None:
    if size > plan["max_single_raw_bytes"]:
        raise BatchError("single_raw_size_budget_exceeded")
    if shutil.disk_usage(root).free < 2 * size + plan["min_free_disk_bytes"]:
        raise BatchError("insufficient_single_object_staging_disk")
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        available = int(next(row.split()[1] for row in meminfo.read_text().splitlines()
                             if row.startswith("MemAvailable:"))) * 1024
        required = max(plan["min_available_ram_bytes"], plan["parser_ram_multiplier"] * size)
        if available < required:
            raise BatchError("insufficient_available_ram_for_serial_native_parse")


def head_object(entry: dict, *, timeout=90, opener=None) -> dict:
    opener = opener or urllib.request.urlopen
    with opener(urllib.request.Request(entry["url"], method="HEAD"), timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        etag, modified = response.headers.get("ETag"), response.headers.get("Last-Modified")
        if response.status != 200 or not isinstance(length, str) or re.fullmatch(r"[1-9][0-9]*", length) is None:
            raise BatchError("official_head_requires_200_and_positive_content_length")
        if not etag or not modified:
            raise BatchError("official_head_etag_and_last_modified_required")
        return {"http_status": 200, "bytes": int(length), "response_headers": {
            "Content-Length": length, "ETag": etag, "Last-Modified": modified},
            "observed_utc": utc_now(), "etag_is_sha256": False}


def same_observed_object(before: dict, after: dict) -> bool:
    return before.get("bytes") == after.get("bytes") and all(
        before.get("response_headers", {}).get(k) == after.get("response_headers", {}).get(k)
        for k in ("ETag", "Last-Modified"))


def download_object(entry: dict, observation: dict, partial: Path, final: Path, *, budget=None, opener=None) -> dict:
    """Append only an exact 206 Content-Range for the freshly observed object."""
    opener = opener or urllib.request.urlopen
    no_symlink_components(partial)
    no_symlink_components(final)
    if final.exists():
        raise BatchError("existing_complete_raw_not_overwritten")
    size = observation["bytes"]
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise BatchError("partial_exceeds_observed_content_length")
    started, downloaded = time.monotonic(), 0
    if offset < size:
        headers = {"If-Match": observation["response_headers"]["ETag"]}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        timeout = budget.timeout() if budget else 90
        transfer = budget.reserve_transfer(size-offset, object_id=entry["object_id"], probe_bytes=1) if budget else None
        eof = False
        try:
            with opener(urllib.request.Request(entry["url"], headers=headers), timeout=timeout) as response:
                if offset:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {offset}-{size-1}/{size}":
                        raise BatchError("resume_requires_exact_206_content_range")
                elif response.status != 200:
                    raise BatchError("initial_get_requires_200")
                if str(response.headers.get("Content-Length")) != str(size-offset):
                    raise BatchError("get_content_length_disagrees_with_head_or_range")
                if any(response.headers.get(k) != observation["response_headers"][k] for k in ("ETag", "Last-Modified")):
                    raise BatchError("get_validators_disagree_with_fresh_head")
                mode = "ab" if partial.exists() else "xb"
                with partial.open(mode) as stream:
                    while True:
                        if budget:
                            budget.check()
                        block = response.read(min(CHUNK, size-offset-downloaded+1))
                        if not isinstance(block, bytes):
                            raise BatchError("non_binary_official_get_body")
                        if not block:
                            eof = True
                            break
                        downloaded += len(block)
                        if budget:
                            budget.consume(len(block), transfer_id=transfer)
                        if offset + downloaded > size:
                            raise BatchError("official_get_body_exceeds_observed_size")
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
        finally:
            if budget:
                budget.finish_transfer(transfer, complete_body_read=eof and downloaded == size-offset)
    sha, actual = sha256_file(partial)
    if actual != size:
        raise BatchError("official_get_incomplete_body")
    # Hard-link commit refuses to overwrite a concurrently introduced original.
    os.link(partial, final)
    partial.unlink()
    return {"bytes": actual, "sha256": sha, "complete_body_read": True,
            "http_status": 200, "response_content_length": size,
            "get_attempted_this_invocation": offset < size,
            "get_http_status_this_invocation": (206 if offset else 200) if offset < size else None,
            "resumed_from_bytes": offset, "download_bytes_this_invocation": downloaded,
            "download_duration_s": time.monotonic()-started, "completed_utc": utc_now(),
            "verification": "complete_local_body_sha256_not_etag", "provider_sha256_verified": False}


def verify_local(path: Path, observation: dict, *, expected_sha=None) -> dict:
    no_symlink_components(path)
    sha, size = sha256_file(path)
    if size != observation["bytes"] or (expected_sha and sha != expected_sha):
        raise BatchError("local_raw_size_or_prior_sha256_mismatch")
    return {"bytes": size, "sha256": sha, "complete_body_read": True,
            "http_status": 200, "response_content_length": size,
            "get_attempted_this_invocation": False, "download_bytes_this_invocation": 0,
            "verification": "complete_local_body_sha256_not_etag", "provider_sha256_verified": False}


def specimen_for(root: Path, entry: dict):
    """Only the explicit retained four-sample inventory grants sample reuse."""
    inventory = root / "audit/source_schema/raw_inventory.json"
    if not inventory.exists():
        return None
    data = json.loads(inventory.read_text())
    if data.get("status") != "downloaded_schema_specimens_only":
        raise BatchError("retained_schema_specimen_inventory_invalid")
    for row in data.get("files", []):
        if (row.get("dataset"), str(row.get("provider_subject")), str(row.get("session_id")), row.get("file")) != identity(entry):
            continue
        expected = f"data/source_schema/{entry['dataset']}/{entry['file']}"
        if (row.get("project_relative_path") != expected or row.get("url") != entry["url"]
                or row.get("http_status") != 200 or row.get("bytes") != row.get("expected_bytes")
                or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", "")))):
            raise BatchError("retained_schema_specimen_identity_or_receipt_invalid")
        return local_path(root, expected), row
    return None


def identity(entry: dict) -> tuple:
    return entry["dataset"], str(entry["provider_subject"]), str(entry["session_id"]), entry["file"]


def validate_entry(entry: dict) -> None:
    targets = expected_source_objects()
    key = identity(entry)
    if (key not in targets or entry.get("url") != targets[key] or entry.get("object_id") != digest(list(key))
            or entry.get("version") != VERSIONS[key[0]]):
        raise BatchError("object_not_exact_registered_source_candidate")


def raw_staging_path(root: Path, plan: dict, entry: dict) -> Path:
    return local_path(root, f"data/source_preparation/staging/{entry['object_id']}/{entry['file']}")


def validate_native(result: dict, entry: dict, raw: dict) -> None:
    rows, summary = result.get("trials"), result.get("summary", {})
    if (not isinstance(rows, list) or not rows or summary.get("source_file_sha256") != raw["sha256"]
            or summary.get("source_file_bytes") != raw["bytes"] or summary.get("n_native_lr_trials") != len(rows)):
        raise BatchError("native_schema_summary_identity_or_count_failed")
    seen = set()
    for row in rows:
        if (row.get("dataset") != entry["dataset"] or row.get("role") != "source"
                or str(row.get("provider_subject")) != str(entry["provider_subject"])
                or str(row.get("session_id")) != str(entry["session_id"])
                or row.get("source_version") != entry["version"]
                or row.get("source_file_sha256") != raw["sha256"] or row.get("source_file_bytes") != raw["bytes"]
                or row.get("canonical_label") not in {"left_hand", "right_hand"}
                or re.fullmatch(r"[0-9a-f]{64}", str(row.get("trial_uid", ""))) is None
                or row["trial_uid"] in seen or row.get("task_end_verified") is not True
                or type(row.get("task_end_sample")) is not int):
            raise BatchError("native_trial_metadata_identity_or_task_end_failed")
        seen.add(row["trial_uid"])


def verify_completed(state: dict, root: Path, *, entry=None, plan=None) -> bool:
    if state.get("status") not in {"COMPLETE", "SCHEMA_BLOCKED", "BACKUP_VERIFIED"}:
        return False
    if (entry and state.get("identity") != list(identity(entry))) or (plan and state.get("plan_sha256") != plan["plan_sha256"]):
        raise BatchError("completed_object_identity_or_plan_mismatch")
    retained = state.get("retained_metadata_sha256", {})
    if not retained or any(file_digest(local_path(root, p)) != sha for p, sha in retained.items()):
        raise BatchError("retained_metadata_integrity_failed")
    receipt = state.get("backup_receipt", {})
    receipt_path = local_path(root, receipt.get("path"))
    if file_digest(receipt_path) != receipt.get("sha256"):
        raise BatchError("backup_receipt_integrity_failed")
    rows, _ = load_verified_receipt(receipt_path)
    manifest = json.loads(local_path(root, state["backup_manifest"]["path"]).read_text())
    fields = ("path", "sha256", "bytes", "role", "dataset")
    expected = {tuple(row[k] for k in fields) for row in manifest["files"]}
    if expected != {tuple(row[k] for k in fields) for row in rows}:
        raise BatchError("backup_receipt_exact_manifest_object_set_failed")
    raw_rows = [r for r in rows if r["role"] == "source_development_raw"]
    if (len(raw_rows) != 1 or raw_rows[0]["path"] != state["raw_path"]
            or raw_rows[0]["sha256"] != state["raw_integrity"]["sha256"]
            or raw_rows[0]["bytes"] != state["raw_integrity"]["bytes"]
            or raw_rows[0]["dataset"] != state["identity"][0]):
        raise BatchError("backup_raw_object_identity_failed")
    return True


def finalize_archived_raw(state, *, root, plan, entry, state_path):
    verify_completed(state, root, entry=entry, plan=plan)
    raw = local_path(root, state["raw_path"])
    if state.get("temporary_raw_owned") is True:
        if raw != raw_staging_path(root, plan, entry):
            raise BatchError("temporary_raw_ownership_path_failed")
        if raw.exists():
            verify_local(raw, state["head_observation"], expected_sha=state["raw_integrity"]["sha256"])
            raw.unlink()
        state["raw_removed_after_verified_backup"] = True
    else:
        if not raw.is_file():
            raise BatchError("unowned_original_missing_must_be_retained")
        verify_local(raw, state["head_observation"], expected_sha=state["raw_integrity"]["sha256"])
        state["raw_removed_after_verified_backup"] = False
        state["existing_raw_preserved"] = True
    state.update(status="SCHEMA_BLOCKED" if state.get("schema_blocked") else "COMPLETE", completed_utc=utc_now())
    atomic_json(state_path, state)
    return state


def recover_interrupted_cache_restore(state, *, root, plan, entry, state_path) -> bool:
    """Reconcile a cache restore left by a process killed before its finally.

    An archived raw reappearing at staging is never enough to prove ownership.
    The cache builder must also prove its durable active marker, the exact
    canonical candidate identity, and the immutable full-GET backup/local SHA.
    All failures preserve the file and become a fixed, non-sensitive code.
    """
    raw = local_path(root, state["raw_path"])
    if not raw.exists() or state.get("raw_removed_after_verified_backup") is not True:
        return False
    if state.get("temporary_raw_owned") is not True or raw != raw_staging_path(root, plan, entry):
        raise BatchError("interrupted_cache_restore_ownership_unverified_raw_preserved")
    item = {**entry, "object_uid": entry["object_id"], "path": state["raw_path"],
            "sha256": state["raw_integrity"]["sha256"], "bytes": state["raw_integrity"]["bytes"],
            "backup_receipt": state["backup_receipt"], "temporary_raw_owned": True,
            "raw_storage": "verified_r2_only"}
    try:
        # Import here to keep the cache restoration policy in its owning module.
        from tools.prepare_source_cache import recover_owned_cache_restore
        recovered = recover_owned_cache_restore(root, item)
        if recovered is not True or raw.exists():
            raise BatchError("interrupted_cache_restore_not_recovered_raw_preserved")
    except Exception:
        raise BatchError("interrupted_cache_restore_ownership_or_integrity_unverified_raw_preserved") from None
    state["recovered_owned_cache_restore"] = True
    state["recovered_owned_cache_restore_utc"] = utc_now()
    atomic_json(state_path, state)
    return True


def process_file(entry, *, root, batch_dir, plan, contract_sha256, budget=None,
                 backup_fn=None, audit_fn=None, opener=None) -> dict:
    validate_entry(entry)
    root = root.resolve(strict=True)
    object_dir = batch_dir / "objects" / entry["object_id"]
    no_symlink_components(object_dir)
    object_dir.mkdir(parents=True, exist_ok=True)
    state_path = object_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "identity": list(identity(entry)), "object_id": entry["object_id"], "plan_sha256": plan["plan_sha256"],
        "input_contract_sha256": contract_sha256, "status": "PENDING", "started_utc": utc_now()}
    if (state.get("plan_sha256") != plan["plan_sha256"] or state.get("identity") != list(identity(entry))
            or state.get("input_contract_sha256") != contract_sha256):
        raise BatchError("object_state_identity_or_contract_mismatch")
    if verify_completed(state, root, entry=entry, plan=plan):
        if state["status"] == "BACKUP_VERIFIED":
            return finalize_archived_raw(state, root=root, plan=plan, entry=entry, state_path=state_path)
        if state["status"] == "COMPLETE":
            recover_interrupted_cache_restore(state, root=root, plan=plan, entry=entry, state_path=state_path)
        return state
    backup_fn, audit_fn = backup_fn or run_backup, audit_fn or audit_native_source
    try:
        # A crash after backup but before terminal status may be replayed without
        # fetching provider bytes again; every retained receipt is revalidated.
        if state.get("backup_receipt") and state.get("retained_metadata_sha256"):
            state["status"] = "BACKUP_VERIFIED"
            if verify_completed(state, root, entry=entry, plan=plan):
                return finalize_archived_raw(state, root=root, plan=plan, entry=entry, state_path=state_path)
        observation = head_object(entry, timeout=budget.timeout() if budget else 90, opener=opener)
        old = state.get("head_observation")
        if old and not same_observed_object(old, observation):
            raise BatchError("provider_object_changed_preserve_prior_partial_and_raw")
        check_space(root, observation["bytes"], plan)
        specimen = specimen_for(root, entry)
        temporary = raw_staging_path(root, plan, entry)
        if specimen:
            raw, prior = specimen
            owned, expected_sha = False, prior["sha256"]
            if not raw.is_file():
                raise BatchError("retained_schema_sample_missing_must_be_preserved")
        else:
            raw, expected_sha = temporary, state.get("raw_integrity", {}).get("sha256")
            partial = raw.with_name(raw.name + ".partial")
            owned = (state.get("temporary_raw_owned") is True and state.get("raw_path") == raw.relative_to(root).as_posix())
            if not raw.exists() and not partial.exists() and "raw_path" not in state:
                owned = True
            if partial.exists() and not old:
                raise BatchError("partial_without_prior_head_provenance_preserved")
        raw.parent.mkdir(parents=True, exist_ok=True)
        state.update(status="HEAD_VERIFIED", head_observation=observation,
                     raw_path=raw.relative_to(root).as_posix(), temporary_raw_owned=owned)
        atomic_json(state_path, state)
        raw_info = (verify_local(raw, observation, expected_sha=expected_sha) if raw.exists() else
                    download_object(entry, observation, raw.with_name(raw.name + ".partial"), raw, budget=budget, opener=opener))
        state.update(status="RAW_VERIFIED", raw_integrity=raw_info)
        atomic_json(state_path, state)
        started = time.monotonic()
        try:
            result = audit_fn(entry["dataset"], raw, provider_subject=entry["provider_subject"],
                              session_id=entry["session_id"], source_version=entry["version"], expected_sha256=raw_info["sha256"])
            validate_native(result, entry, raw_info)
            rows, audit = result["trials"], result["summary"]
            state["schema_blocked"] = False
        except Exception as exc:
            state["schema_blocked"] = True
            rows = []
            audit = {"status": "SCHEMA_BLOCKED", "dataset": entry["dataset"], "identity": list(identity(entry)),
                     "source_file_sha256": raw_info["sha256"], "source_file_bytes": raw_info["bytes"],
                     "error_type": type(exc).__name__, "error_code": str(exc) if isinstance(exc, BatchError) else "native_schema_details_suppressed",
                     "identity_channel_task_qualification": None}
        audit_path, metadata_path = object_dir / "adapter_audit.json", object_dir / "trial_metadata.jsonl"
        atomic_json(audit_path, audit)
        # Keep every original native metadata row unchanged, including task end.
        with metadata_path.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        state.update(status="METADATA_EXTRACTED", native_schema_audited=not state["schema_blocked"],
                     parse_duration_s=time.monotonic()-started, native_trial_count=len(rows),
                     adapter_audit=descriptor(root, audit_path), native_trials=descriptor(root, metadata_path))
        atomic_json(state_path, state)
        manifest_path, receipt_path = object_dir / "backup_manifest.json", object_dir / "backup_receipt.json"
        manifest = {"schema_version": 1, "files": []}
        for path, role in ((raw, "source_development_raw"), (audit_path, "audit"), (metadata_path, "audit")):
            sha, size = sha256_file(path)
            manifest["files"].append({"path": path.relative_to(root).as_posix(), "sha256": sha,
                                      "bytes": size, "role": role, "dataset": entry["dataset"]})
        atomic_json(manifest_path, manifest)
        if budget:
            budget.check()
        prefix = f"{PREFIX}{plan['plan_sha256']}/{entry['object_id']}/"
        r2_transfer = budget.reserve_transfer(sum(item["bytes"] for item in manifest["files"]),
            object_id=entry["object_id"], scope=ResourceBudget.R2_BACKUP, probe_bytes=len(manifest["files"])) if budget else None
        backup = None
        instrumented = budget is not None and backup_fn is _NATIVE_BACKUP_IMPLEMENTATION
        try:
            options = {}
            if instrumented:
                options["client"] = BudgetedR2Client(create_client(), budget, r2_transfer,
                    {prefix + item["path"]: item["bytes"] for item in manifest["files"]}, scope=ResourceBudget.R2_BACKUP)
            backup = backup_fn(manifest_path=manifest_path, root=root, prefix=prefix, receipt_path=receipt_path, execute=True, **options)
        finally:
            if budget:
                try:
                    if isinstance(backup, dict) and not instrumented:
                        observed = sum(item.get("get", {}).get("bytes_read", 0) for item in backup.get("objects", []))
                        budget.consume(observed, transfer_id=r2_transfer, scope=ResourceBudget.R2_BACKUP)
                finally:
                    budget.finish_transfer(r2_transfer, complete_body_read=isinstance(backup, dict) and
                        backup.get("status") == "verified", scope=ResourceBudget.R2_BACKUP)
        if backup.get("status") != "verified" or backup.get("execute") is not True:
            raise BatchError("r2_put_full_get_sha256_not_verified")
        state.update(status="BACKUP_VERIFIED", backup_manifest=descriptor(root, manifest_path),
                     backup_receipt=descriptor(root, receipt_path), r2_prefix=prefix,
                     r2_full_get_bytes_verified=sum(r.get("get", {}).get("bytes_read", 0) for r in backup.get("objects", [])),
                     retained_metadata_sha256={p.relative_to(root).as_posix(): file_digest(p)
                         for p in (audit_path, metadata_path, manifest_path, receipt_path)})
        atomic_json(state_path, state)
        return finalize_archived_raw(state, root=root, plan=plan, entry=entry, state_path=state_path)
    except Exception as exc:
        state.update(status="FAILED", failed_utc=utc_now(), error_type=type(exc).__name__,
                     error_code=str(exc) if isinstance(exc, BatchError) else "operation_failed_details_suppressed")
        atomic_json(state_path, state)
        raise BatchError(state["error_code"]) from None


def aggregate_inventory(plan, *, root, batch_dir, budget=None) -> dict:
    """Full raw inventory only after all 505 actual native audits and receipts.

    A complete raw archive can have R2-only raw bodies. This is not a numerical
    cache nor W2 completion; a later worker must explicitly restore staging.
    """
    validate_plan(plan)
    root = root.resolve(strict=True)
    objects, missing, blocked, seen = [], [], [], set()
    for entry in plan["files"]:
        if budget:
            budget.check()
        state_path = batch_dir / "objects" / entry["object_id"] / "state.json"
        if not state_path.exists():
            missing.append(entry["object_id"])
            continue
        state = json.loads(state_path.read_text())
        if not verify_completed(state, root, entry=entry, plan=plan) or state.get("status") != "COMPLETE" or state.get("native_schema_audited") is not True:
            blocked.append(entry["object_id"])
            continue
        rows = [json.loads(line) for line in local_path(root, state["native_trials"]["path"]).read_text().splitlines() if line]
        validate_native({"trials": rows, "summary": json.loads(local_path(root, state["adapter_audit"]["path"]).read_text())}, entry, state["raw_integrity"])
        for row in rows:
            if row["trial_uid"] in seen:
                raise BatchError("repeated_original_trial_uid_across_source_objects")
            seen.add(row["trial_uid"])
        raw = local_path(root, state["raw_path"])
        removed = state.get("raw_removed_after_verified_backup") is True
        if removed:
            if state.get("temporary_raw_owned") is not True or raw != raw_staging_path(root, plan, entry):
                raise BatchError("archived_raw_removal_ownership_or_presence_failed")
            recover_interrupted_cache_restore(state, root=root, plan=plan, entry=entry, state_path=state_path)
            if raw.exists():
                raise BatchError("archived_raw_removal_ownership_or_presence_failed")
        else:
            verify_local(raw, state["head_observation"], expected_sha=state["raw_integrity"]["sha256"])
        objects.append({k: entry[k] for k in ("dataset", "provider_subject", "session_id", "file", "version", "url")})
        objects[-1].update(object_uid=entry["object_id"], path=state["raw_path"], sha256=state["raw_integrity"]["sha256"], bytes=state["raw_integrity"]["bytes"],
            complete_download=True, http_status=200, response_content_length=str(state["head_observation"]["bytes"]),
            native_schema_audited=True, native_trials=state["native_trials"], adapter_audit=state["adapter_audit"],
            backup_receipt=state["backup_receipt"], temporary_raw_owned=state["temporary_raw_owned"],
            raw_removed_after_verified_backup=removed, raw_storage="verified_r2_only" if removed else "verified_local_and_r2",
            head_observation=state["head_observation"], raw_download_receipt=state["raw_integrity"])
    complete = len(objects) == 505 and not missing and not blocked
    evidence = root / "audit/source_schema/evidence_manifest.json"
    if not evidence.is_file():
        raise BatchError("retained_provider_terms_evidence_required")
    for entry in json.loads(evidence.read_text()).get("files", []):
        if file_digest(local_path(root, entry["path"])) != entry["sha256"]:
            raise BatchError("provider_terms_evidence_integrity_failed")
    result = {"schema_version": "paper3-source-native-inventory-v2",
        "status": "SOURCE_RAW_INVENTORY_COMPLETE" if complete else "SOURCE_RAW_INVENTORY_PARTIAL",
        "role": "source_train", "synthetic": False, "old_npz_reused": False,
        "provider_roster_verified": complete, "required_target_object_count": 505,
        "actual_verified_objects": len(objects), "required_objects_not_yet_verified": 505-len(objects),
        "missing_object_ids": missing, "blocked_object_ids": blocked, "files": objects,
        "provider_terms_evidence": descriptor(root, evidence), "candidate_plan_sha256": plan["plan_sha256"],
        "numerical_cache_created": False, "training_started": False, "w2_complete": False,
        "next_step": "explicit restore of verified R2-only raw staging for prepare_source_cache.py"}
    atomic_json(batch_dir / "source_native_inventory.json", result)
    return result


def execute_plan(plan, *, root=REPO, batch_dir=None, max_new_files=None, process_fn=None,
                 contract_path="contracts/input_contract.json", operational_limits=None,
                 operational_amendment_note=None) -> dict:
    validate_plan(plan)
    root = Path(root).resolve(strict=True)
    contract_sha = execution_gate(root, contract_path)  # Before directories, HTTP, or raw access.
    if max_new_files is not None and (type(max_new_files) is not int or max_new_files < 1):
        raise BatchError("max_new_files_must_be_positive")
    batch_dir = batch_dir or root / "audit/source_preparation/batches" / plan["plan_sha256"]
    no_symlink_components(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    process_fn = process_fn or process_file
    # The staging namespace is shared across plan revisions. A global worker
    # lock prevents concurrent plans from touching the same candidate object.
    lock_path = root / "audit/source_preparation/.source_worker.lock"
    no_symlink_components(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BatchError("source_batch_already_active") from None
        operational_limits = operational_limits or {}
        if set(operational_limits) - {"max_download_bytes", "max_wall_seconds", "max_r2_readback_bytes"}:
            raise BatchError("unknown_operational_resource_limit")
        identity_sha = digest({"schema": "source-raw-resource-identity-v2", "files": plan["files"],
            "adapter_code_sha256": plan["adapter_code_sha256"], "input_contract_sha256": contract_sha})
        budget = ResourceBudget(plan, root / "audit/source_preparation/resource_progress.json",
            identity_sha256=identity_sha, amendment_note=operational_amendment_note, **operational_limits)
        try:
            return _execute_serial_candidates(plan, root, batch_dir, max_new_files, process_fn, contract_sha, budget)
        finally:
            budget.close()


def _execute_serial_candidates(plan, root, batch_dir, max_new_files, process_fn, contract_sha, budget):
        counts = {"new_objects_attempted": 0, "archived_schema_verified": 0, "schema_blocked": 0, "failed": 0}
        status = "ALL_CANDIDATES_ATTEMPTED"
        for entry in plan["files"]:
            try:
                budget.check()
            except BatchError as exc:
                status = str(exc).upper() + "_NO_STAGE_ADVANCE"
                break
            state_path = batch_dir / "objects" / entry["object_id"] / "state.json"
            prior = json.loads(state_path.read_text()) if state_path.exists() else {}
            try:
                completed_before = verify_completed(prior, root, entry=entry, plan=plan)
                if not completed_before:
                    if max_new_files is not None and counts["new_objects_attempted"] >= max_new_files:
                        status = "PARTIAL_BATCH_LIMIT_NO_STAGE_ADVANCE"
                        break
                    counts["new_objects_attempted"] += 1
                state = process_fn(entry, root=root, batch_dir=batch_dir, plan=plan,
                                   contract_sha256=contract_sha, budget=budget)
                counts["archived_schema_verified" if state["status"] == "COMPLETE" else "schema_blocked"] += 1
            except Exception:
                # Fixed-code failure evidence belongs to the isolated object.
                # Continue other independent candidates within remaining budget.
                counts["failed"] += 1
            finally:
                budget.checkpoint()
            atomic_json(batch_dir / "batch_progress.json", {"status": "RUNNING", **counts,
                "plan_sha256": plan["plan_sha256"], "last_object_id": entry["object_id"]})
        try:
            result = aggregate_inventory(plan, root=root, batch_dir=batch_dir, budget=budget)
        except BatchError as error:
            status = str(error).upper() + "_NO_STAGE_ADVANCE"
            result = {"status": "SOURCE_RAW_INVENTORY_PARTIAL", "actual_verified_objects": None}
        summary = {"status": status, "inventory_status": result["status"], **counts,
                   "actual_verified_objects": result["actual_verified_objects"], "required_target_object_count": 505,
                   "download_bytes_read": budget.data["download_bytes_read"],
                   "download_bytes_charged": budget.data["download_bytes_charged"],
                   "byte_scopes": {name: {key: ledger[key] for key in ("observed_bytes", "charged_bytes", "limit_bytes")}
                                   for name, ledger in budget.data["byte_scopes"].items()},
                   "resource_progress_path": budget.path.relative_to(root).as_posix(),
                   "active_wall_elapsed_s": budget.data["active_wall_elapsed_s"],
                   "numerical_cache_created": False, "training_started": False, "w2_complete": False}
        atomic_json(batch_dir / "batch_progress.json", summary)
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--write-plan", type=Path, help="Write candidate metadata only; never verifies existence")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-new-files", type=int)
    parser.add_argument("--max-download-bytes", type=int, default=100 * 1024**3)
    parser.add_argument("--max-single-raw-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--max-wall-seconds", type=int, default=86400)
    parser.add_argument("--amend-max-download-bytes", type=int)
    parser.add_argument("--amend-max-wall-seconds", type=int)
    parser.add_argument("--amend-max-r2-readback-bytes", type=int)
    parser.add_argument("--operational-amendment-note")
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text()) if args.plan else build_plan(
            max_download_bytes=args.max_download_bytes, max_single_raw_bytes=args.max_single_raw_bytes,
            max_wall_seconds=args.max_wall_seconds)
        validate_plan(plan)
        if args.write_plan:
            atomic_json(args.write_plan, plan)
        summary = {"status": "CANDIDATE_PLAN_ONLY", "execute": args.execute,
                   **{k: plan[k] for k in ("plan_sha256", "candidate_count", "provider_roster_verified",
                      "inflight_downloads", "parallel_native_parsers", "max_download_bytes", "max_single_raw_bytes", "max_wall_seconds")},
                   "numerical_cache_created": False, "training_started": False}
        print(json.dumps(summary), flush=True)
        if args.execute:
            limits = {name: value for name, value in {
                "max_download_bytes": args.amend_max_download_bytes, "max_wall_seconds": args.amend_max_wall_seconds,
                "max_r2_readback_bytes": args.amend_max_r2_readback_bytes}.items() if value is not None}
            result = execute_plan(plan, max_new_files=args.max_new_files, operational_limits=limits,
                                  operational_amendment_note=args.operational_amendment_note)
            print(json.dumps(result), flush=True)
            return 0 if result.get("inventory_status") == "SOURCE_RAW_INVENTORY_COMPLETE" else 2
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED_CLOSED", "error_type": type(exc).__name__,
                          "error_code": str(exc) if isinstance(exc, BatchError) else "operation_failed_details_suppressed"}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
