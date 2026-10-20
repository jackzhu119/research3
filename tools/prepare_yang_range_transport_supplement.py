#!/usr/bin/env python3
"""One explicitly authorized Yang record-preparation supplement, never finalize.

Frozen scientific code alone parses, checks and caches each record. This new
entry adds bounded logical transport retries, charged to the original ledger.
It never calls prepare/assemble, fits a model or writes coordinator COMPLETE.
The unchanged V2 call07 must later produce the real final five-object GET proof.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import resume_yang_source_transport_v2 as parent
from p3work import source_extension_coordinator as core
from p3work import source_extension_io as storage

SCHEMA = "paper3-yang-one-record-preparation-transport-supplement-v1"
ENTRY = "tools/prepare_yang_range_transport_supplement.py"
TESTS = "tests/test_prepare_yang_range_transport_supplement.py"
BASE = "audit/data_plan_20261020/yang_range_transport_supplement_v1"
AMENDMENT = "plans/data_plan_20261019/yang_range_transport_supplement_amendment.json"
PLAN_SHA = parent.PLAN_SHA
OFFICIAL_URL = "https://ndownloader.figshare.com/files/51001884"
PARENT_SHA = "2ae1e75e47a75032155da44c1cb45afa78a1ec82da3ecd11d6f85ec60e1c0f71"
PARENT_STATE_SHA = "8b325af72e8420b07b8416cc2e45fe359353b755ccac8605283cfb6ebbff095e"
PARENT_RECEIPT_SHA = "a3af5de3aa9ab6f3da964dc35549e0260c1f48f5fe93d65245f59bd585ff8478"
PARENT_AFTER_SHA = "4584e04f594704ad7e151a28d1ffaf95ac86a35df5d603b943afcdb033bca43b"
INITIAL_COMPLETE = 82
RECORD_COUNT = 153
MAX_ATTEMPTS = 4
MAX_EXTRA_RETRIES = 64
RETRY_STATUSES = frozenset({429, 502, 503, 504})
BACKOFF = (1, 2, 4)
AUTHORIZATION = {
    "authorized_by": "root_20261020_under_user_Paper3_CPU_preparation_authorization",
    "scope": "one_additional_bounded_record_preparation_invocation_no_global_finalization",
    "maximum_supplement_invocations": 1,
    "original_V2_calls_already_used": 6, "original_V2_call_limit_unchanged": 8,
    "original_scientific_plan_URL_CRC_SHA_native_signals_resources_clock_unchanged": True,
    "no_fit_GPU_RunPod_external_query_or_credential_change": True,
}


class SupplementError(ValueError):
    pass


read = parent.read
path = parent.path
ref = parent.ref
checked = parent.checked
immutable_bytes = parent.immutable_bytes


def ledger_file(root):
    return parent.previous.ledger_path(root, {"plan_sha256": PLAN_SHA})


def audit_dir(root):
    return path(root, f"audit/source_extensions_20261019/Yang2025/{PLAN_SHA}")


def closed(root, plan, filename=None):
    value, body = parent.closed_ledger(root, plan, filename)
    if "wall_intent" in value or "active_transfer" in value:
        raise SupplementError("closed_original_ledger_must_have_no_intent_keys")
    return value, body


def code(root):
    filename = path(root, ENTRY)
    if filename.read_bytes() != Path(__file__).read_bytes():
        raise SupplementError("executed_supplement_differs_from_root")
    return ref(root, filename)


def load(root):
    root, plan, plan_ref = parent.load_plan(root)
    if (ref(root, path(root, parent.ENTRY))["sha256"] != PARENT_SHA
            or plan.get("dataset") != "Yang2025" or len(plan["records"]) != RECORD_COUNT):
        raise SupplementError("exact_frozen_Yang_V2_and_plan_required")
    if any(item["download"]["kind"] != "zip_range" or item["download"]["url"] != OFFICIAL_URL
           for record in plan["records"] for item in record["raw_files"]):
        raise SupplementError("only_original_official_Yang_zip_ranges_allowed")
    parent.verify_static_authority(root, plan)
    return root, plan, plan_ref


def snapshot(root, source, destination):
    immutable_bytes(path(root, destination), Path(source).read_bytes())
    return ref(root, path(root, destination))


def record_states(root, plan, *, require_all=False):
    """Original record/GET/SHA verifier only; no mutable Ledger construction."""
    verified, existing, cache_bytes = [], [], 0
    for record in plan["records"]:
        filename = path(root, f"audit/source_extensions_20261019/Yang2025/{PLAN_SHA}/{record['record_id']}/state.json")
        if not filename.exists():
            if require_all:
                raise SupplementError("all153_original_record_states_required")
            continue
        state = read(filename)
        if state.get("record_id") != record["record_id"] or state.get("plan_sha256") != PLAN_SHA:
            raise SupplementError("original_record_state_identity_changed")
        existing.append((record, filename, state))
        if state.get("status") == "COMPLETE":
            if not core._state_verified(root, state, plan, record):
                raise SupplementError("complete_original_record_not_verified")
            verified.append(record["record_id"])
            cache_bytes += sum(item["bytes"] for item in state["artifacts"])
        elif require_all:
            raise SupplementError("all153_original_records_must_be_complete")
    if cache_bytes > plan["budgets"]["cache_bytes"]:
        raise SupplementError("original_cumulative_cache_budget_exhausted")
    if require_all and len(verified) != RECORD_COUNT:
        raise SupplementError("all153_original_record_states_required")
    return verified, existing, cache_bytes


def starting_parent(root, plan):
    state_path = path(root, parent.HISTORY + "/recovery_ledger.json")
    if ref(root, state_path)["sha256"] != PARENT_STATE_SHA:
        raise SupplementError("exact_closed_V2_call06_state_required_no_new_calls")
    _, amendment_ref = parent.verify_static_authority(root, plan)
    result = parent._history(root, plan, amendment_ref)
    if result is None:
        raise SupplementError("actual_six_V2_calls_required")
    _, _, state, last = result
    if (state["calls_used"] != 6 or state["active_call"] is not None or state["terminal"] is not None
            or len(state["call_receipts"]) != 6):
        raise SupplementError("exact_closed_V2_call06_state_required_no_new_calls")
    receipt_ref = state["call_receipts"][-1]
    receipt = read(checked(root, receipt_ref))
    if (receipt_ref["sha256"] != PARENT_RECEIPT_SHA or receipt.get("call_number") != 6
            or receipt.get("error_type") != "PreparationError"
            or receipt.get("error_code") != "provider_did_not_honor_exact_range"
            or receipt.get("retryable_transport_error") is not True
            or receipt["after_closed_ledger"]["sha256"] != PARENT_AFTER_SHA):
        raise SupplementError("exact_actual_retryable_call06_receipt_required")
    current, body = closed(root, plan)
    if hashlib.sha256(body).hexdigest() != PARENT_AFTER_SHA or current != last:
        raise SupplementError("original_ledger_must_equal_exact_closed_call06_after")
    return state_path, receipt_ref, current


def policy():
    return {"per_range_GET_or_exact_manifest_backup_logical_attempt_limit": MAX_ATTEMPTS,
            "shared_additional_logical_retry_limit": MAX_EXTRA_RETRIES,
            "retry_HTTP_statuses": sorted(RETRY_STATUSES), "backoff_seconds": list(BACKOFF),
            "provider_HTTP_adapter_retries": 0, "R2_SDK_total_max_attempts": 1,
            "provider_trust_env": "requests_default_preserved",
            "does_not_bound_redirect_or_manifest_multiobject_HTTP_request_count": True,
            "every_failed_retry_consumes_full_original_reservation": True}


def write_amendment(root):
    root, plan, plan_ref = load(root)
    with parent.owner(root), parent.previous.coordinator_lock(root):
        target = path(root, AMENDMENT)
        if target.exists():
            _, descriptor = verify_amendment(root, plan)
            return {"status": "AMENDMENT_ALREADY_VERIFIED_NO_EXECUTION", "amendment": descriptor}
        state_path, receipt_ref, _ = starting_parent(root, plan)
        verified, existing, _ = record_states(root, plan)
        if len(verified) != INITIAL_COMPLETE:
            raise SupplementError("exact82_initial_verified_complete_records_required")
        originals = []
        for record, filename, state in existing:
            item = snapshot(root, filename, f"{BASE}/initial_states/{record['record_id']}.json.txt")
            originals.append({"record_id": record["record_id"], "source_at_snapshot": filename.relative_to(root).as_posix(),
                              "status_at_snapshot": state["status"], "opaque_snapshot": item})
        value = {"schema_version": SCHEMA, "status": "AUTHORIZED_ONE_RECORD_PREPARATION_SUPPLEMENT",
            "authorization": AUTHORIZATION, "implementation": code(root), "tests": ref(root, path(root, TESTS)),
            "frozen_plan": plan_ref, "V2_wrapper": ref(root, path(root, parent.ENTRY)),
            "V2_amendment": ref(root, path(root, parent.AMENDMENT_PATH)),
            "actual_V2_call06_receipt": receipt_ref,
            "actual_V2_closed_call_receipts": read(state_path)["call_receipts"],
            "V2_state_opaque": snapshot(root, state_path, BASE + "/initial_V2_state.json.txt"),
            "initial_closed_ledger": snapshot(root, ledger_file(root), BASE + "/initial_closed_ledger.json"),
            "initial_verified_complete_record_ids": verified, "initial_state_inventory": originals,
            "transport_policy": policy(), "global_finalization_allowed": False,
            "finalization_owner": "unchanged_original_Yang_V2_explicit_call07_or_call08",
            "training_started": False, "GPU_actions": 0, "external_payload_accessed": False}
        core.atomic_json(target, value, immutable=True)
        return {"status": value["status"], "amendment": ref(root, target), "execution_started": False}


def verify_amendment(root, plan):
    value = read(path(root, AMENDMENT))
    if (value.get("schema_version") != SCHEMA or value.get("status") != "AUTHORIZED_ONE_RECORD_PREPARATION_SUPPLEMENT"
            or value.get("authorization") != AUTHORIZATION or value.get("implementation") != code(root)
            or value.get("tests") != ref(root, path(root, TESTS)) or value.get("transport_policy") != policy()
            or value.get("frozen_plan") != ref(root, path(root, parent.PLAN_PATH))
            or value.get("V2_wrapper") != ref(root, path(root, parent.ENTRY))
            or value.get("V2_wrapper", {}).get("sha256") != PARENT_SHA
            or value.get("V2_amendment") != ref(root, path(root, parent.AMENDMENT_PATH))
            or value.get("global_finalization_allowed") is not False):
        raise SupplementError("reviewed_supplement_authority_changed")
    receipt = read(checked(root, value["actual_V2_call06_receipt"]))
    state = read(checked(root, value["V2_state_opaque"]))
    initial, _ = closed(root, plan, checked(root, value["initial_closed_ledger"]))
    if (value["actual_V2_call06_receipt"]["sha256"] != PARENT_RECEIPT_SHA
            or value["V2_state_opaque"]["sha256"] != PARENT_STATE_SHA
            or value["initial_closed_ledger"]["sha256"] != PARENT_AFTER_SHA
            or state.get("calls_used") != 6 or state.get("active_call") is not None or state.get("terminal") is not None
            or state.get("call_receipts", [])[-1:] != [value["actual_V2_call06_receipt"]]
            or state.get("call_receipts") != value.get("actual_V2_closed_call_receipts")
            or receipt.get("after_closed_ledger", {}).get("sha256") != PARENT_AFTER_SHA):
        raise SupplementError("frozen_call06_ancestry_changed")
    for descriptor in value["actual_V2_closed_call_receipts"]:
        historical = read(checked(root, descriptor))
        for key in ("intent", "before_closed_ledger", "after_closed_ledger"):
            if key in historical:
                checked(root, historical[key])
    ids = []
    expected = {record["record_id"]: record for record in plan["records"]}
    seen = set()
    for item in value.get("initial_state_inventory", []):
        rid = item.get("record_id")
        if rid not in expected or rid in seen:
            raise SupplementError("initial_state_inventory_identity_changed")
        seen.add(rid)
        old_state = read(checked(root, item["opaque_snapshot"]))
        if (old_state.get("record_id") != rid or old_state.get("plan_sha256") != PLAN_SHA
                or old_state.get("status") != item.get("status_at_snapshot")):
            raise SupplementError("initial_state_inventory_identity_changed")
        if old_state.get("status") == "COMPLETE":
            if not core._state_verified(root, old_state, plan, expected[rid]):
                raise SupplementError("initial82_GET_and_cache_proofs_required")
            ids.append(rid)
    if ids != value.get("initial_verified_complete_record_ids") or len(ids) != INITIAL_COMPLETE:
        raise SupplementError("exact82_initial_verified_complete_records_required")
    return value, ref(root, path(root, AMENDMENT))


def active_reservation(ledger, *, kind, size=None, identity=None):
    value = ledger.data.get("active_transfer")
    if (not isinstance(value, dict) or set(value) != {"kind", "reserved_bytes", "identity"}
            or value["kind"] != kind or type(value["reserved_bytes"]) is not int
            or value["reserved_bytes"] < 0
            or (size is not None and value["reserved_bytes"] != size)
            or (identity is not None and value["identity"] != identity)):
        raise SupplementError("exact_original_active_reservation_required")
    return dict(value)


class RetryBudget:
    """Durable logical counts; this is not a count of redirect/SDK HTTP calls."""
    def __init__(self, root, amendment_ref, *, sleep=time.sleep):
        self.root, self.amendment_ref, self.sleep = Path(root), amendment_ref, sleep
        self.filename = path(root, BASE + "/retry_ledger.json")
        if self.filename.exists():
            raise SupplementError("issued_retry_history_never_reset_or_replayed")
        self.data = {"schema_version": SCHEMA, "amendment": amendment_ref, "logical_attempts": {},
                     "extra_logical_retries": 0, "retry_receipts": [], "active_retry": None,
                     "IO_receipt_snapshots": []}
        self.save()

    def save(self):
        core.atomic_json(self.filename, self.data)

    def begin(self, key):
        if key in self.data["logical_attempts"]:
            raise SupplementError("duplicate_logical_dispatch_not_a_new_retry_allowance")
        self.data["logical_attempts"][key] = 1
        self.save()

    def keep_receipt(self, source, destination):
        descriptor = snapshot(self.root, source, destination)
        self.data["IO_receipt_snapshots"].append({"source_at_snapshot": Path(source).relative_to(self.root).as_posix(),
                                                   "opaque_snapshot": descriptor})
        self.save()
        return descriptor

    def available(self, key):
        return (self.data["active_retry"] is None and self.data["logical_attempts"][key] < MAX_ATTEMPTS
                and self.data["extra_logical_retries"] < MAX_EXTRA_RETRIES)

    def retry(self, key, *, ledger, failure, reservation):
        if not self.available(key):
            raise SupplementError("bounded_logical_retry_allowance_exhausted")
        if failure.get("http_status") not in RETRY_STATUSES:
            raise SupplementError("only_actual_whitelisted_HTTP_status_may_retry")
        if active_reservation(ledger, kind=reservation["kind"]) != reservation:
            raise SupplementError("original_reservation_changed_before_retry")
        sequence = self.data["extra_logical_retries"] + 1
        delay = BACKOFF[self.data["logical_attempts"][key] - 1]
        if not 0 <= delay <= 30:
            raise SupplementError("backoff_must_be_at_most30_seconds")
        index = len(ledger.data["history"])
        intent = {"schema_version": SCHEMA, "amendment": self.amendment_ref, "logical_key": key,
                  "extra_retry_number": sequence, "next_logical_attempt": self.data["logical_attempts"][key] + 1,
                  "failure": failure, "original_reservation": reservation,
                  "failed_original_history_index": index, "backoff_seconds": delay}
        location = path(self.root, f"{BASE}/retry-{sequence:03d}")
        core.atomic_json(location / "intent.json", intent, immutable=True)
        self.data["extra_logical_retries"] = sequence
        self.data["logical_attempts"][key] += 1
        self.data["active_retry"] = sequence
        self.save()  # Durable allowance is consumed BEFORE another request.
        ledger.finish()  # Never provide verified_bytes for a failed attempt.
        expected = {**reservation, "verified_bytes": None, "status": "FAILURE_FULL_RESERVATION_CONSUMED"}
        if ledger.data["history"][index] != expected:
            raise SupplementError("failed_retry_must_consume_full_original_reservation")
        self.sleep(delay)
        ledger.reserve(reservation["kind"], reservation["reserved_bytes"], reservation["identity"])
        value = {"schema_version": SCHEMA, "intent": ref(self.root, location / "intent.json"),
                 "failed_original_history_index": index, "failed_transfer": expected,
                 "next_original_reservation": active_reservation(ledger, kind=reservation["kind"])}
        core.atomic_json(location / "receipt.json", value, immutable=True)
        self.data["retry_receipts"].append(ref(self.root, location / "receipt.json"))
        self.data["active_retry"] = None
        self.save()


def safe_response(response):
    values = {"http_status": response.status_code}
    # Only primitive range facts, never URLs, request IDs or arbitrary headers.
    for name, pattern in (("Content-Length", r"\d{1,20}"),
                          ("Content-Range", r"bytes \d{1,20}-\d{1,20}/\d{1,20}"),
                          ("Content-Encoding", r"identity|gzip|br|deflate")):
        value = response.headers.get(name)
        values[name] = value if isinstance(value, str) and re.fullmatch(pattern, value) else None
    return values


class RangeSession:
    def __init__(self, session, ledger, item, budget):
        self.session, self.ledger, self.item, self.budget = session, ledger, item, budget

    def get(self, url, **kwargs):
        headers = kwargs.get("headers")
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", str((headers or {}).get("Range", "")))
        if (url != OFFICIAL_URL or url != self.item["download"]["url"] or not match
                or headers != {"Range": match.group(0), "Accept-Encoding": "identity"}
                or kwargs != {"headers": headers, "stream": True, "timeout": (20, 120)}
                or self.session.trust_env is not True
                or any(self.session.headers.get(k) != v for k, v in parent.HEADERS.items())
                or self.session.adapters["https://"].max_retries.total != 0):
            raise SupplementError("only_exact_original_NoCache_identity_range_request_allowed")
        start, end = map(int, match.groups())
        length = end - start + 1
        if start < 0 or length <= 0 or length > core.RANGE_CHUNK or end >= self.item["download"]["archive_bytes"]:
            raise SupplementError("range_outside_original_archive_or_chunk_bound")
        identity = f"range:{start}:{length}"
        reservation = active_reservation(self.ledger, kind="provider_body_bytes", size=length, identity=identity)
        key = core.digest({"kind": "original_provider_range_GET", "raw_path": self.item["path"], "identity": identity})
        self.budget.begin(key)
        while True:
            response = self.session.get(url, **kwargs)
            if response.status_code not in RETRY_STATUSES or not self.budget.available(key):
                return response  # Original _range owns every header/body/CRC/SHA check.
            failure = safe_response(response)
            response.close()  # Do not consume or treat an error body as source data.
            self.budget.retry(key, ledger=self.ledger, failure=failure, reservation=reservation)


def download_with_retries(item, target, *, session, ledger, budget):
    if item["download"]["kind"] != "zip_range" or item["download"]["url"] != OFFICIAL_URL:
        raise SupplementError("supplement_download_must_be_original_Yang_range")
    return core.download_component(item, target, session=RangeSession(session, ledger, item, budget), ledger=ledger)


def create_r2_client():
    """Existing environment only, matching original timeouts with SDK retry off."""
    import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT"], region_name="auto",
                        config=Config(connect_timeout=15, read_timeout=120,
                                      retries={"total_max_attempts": 1, "mode": "standard"}))


def retryable_backup_failure(result, *, manifest_sha, prefix, files):
    if (result.get("status") != "failed" or result.get("execute") is not True
            or result.get("source_extension_profile") != storage.profile()
            or result.get("manifest_sha256") != manifest_sha or result.get("prefix") != prefix
            or result.get("error_type") != "BackupError" or result.get("error_code") != "object_backup_not_verified"):
        return None
    allowed = {item["path"]: item for item in files}
    failed, seen = [], set()
    for row in result.get("objects", []):
        name = row.get("path")
        if name not in allowed or name in seen or any(row.get(k) != allowed[name][k] for k in storage.ENTRY_FIELDS):
            return None
        seen.add(name)
        if row.get("status") == "verified":
            get = row.get("get", {})
            if (row.get("put", {}).get("status") not in {"created", "already_exists_not_overwritten"}
                    or get.get("status") != "verified" or get.get("http_status") != 200
                    or get.get("complete_body_read") is not True or get.get("bytes_read") != row["bytes"]
                    or get.get("sha256") != row["sha256"]):
                return None
            continue
        if row.get("status") != "failed":
            return None
        put, get = row.get("put", {}), row.get("get", {})
        outcome = get if put.get("status") in {"created", "already_exists_not_overwritten"} else put
        if (outcome.get("status") != "failed" or outcome.get("http_status") not in RETRY_STATUSES
                or outcome.get("error_type") not in {"ClientError", "BackupError"}
                or (outcome.get("error_type") == "BackupError"
                    and outcome.get("error_code") not in {"put_http_status_not_200", "get_http_status_not_200"})):
            return None
        failed.append({"path": name, "http_status": outcome["http_status"], "operation": "get" if outcome is get else "put"})
    return failed if failed else None


class BackupWithRetries:
    def __init__(self, root, ledger, budget):
        self.root, self.ledger, self.budget, self.client = Path(root), ledger, budget, None

    def __call__(self, *, manifest_path, root, prefix, receipt_path, execute):
        if Path(root) != self.root or execute is not True:
            raise SupplementError("only_original_executing_backup_allowed")
        manifest_path = path(self.root, Path(manifest_path).relative_to(self.root).as_posix())
        receipt_path = path(self.root, Path(receipt_path).relative_to(self.root).as_posix())
        base = f"audit/source_extensions_20261019/Yang2025/{PLAN_SHA}/"
        if (not manifest_path.relative_to(root).as_posix().startswith(base)
                or not receipt_path.relative_to(root).as_posix().startswith(base)
                or not prefix.startswith(f"paper3/source-extensions-20261019/Yang2025/{PLAN_SHA}/")
                or not prefix.endswith(("/raw/", "/cache/"))):
            raise SupplementError("backup_outside_original_record_namespace")
        original, _ = storage._modules()
        files, manifest_sha = original.load_manifest(manifest_path, self.root)
        if any(item["dataset"] != "Yang2025" for item in files):
            raise SupplementError("backup_only_original_Yang_rows_allowed")
        reserve = sum(item["bytes"] for item in files)
        reservation = active_reservation(self.ledger, kind="r2_get_body_bytes", size=reserve, identity=prefix)
        key = core.digest({"kind": "original_manifest_backup_call", "manifest_sha256": manifest_sha, "prefix": prefix})
        self.budget.begin(key)
        attempt = 0
        while True:
            attempt += 1
            if receipt_path.exists():
                self.budget.keep_receipt(receipt_path, f"{BASE}/backup-history/{key}/before-{attempt:02d}.json.txt")
            if self.client is None:
                self.client = create_r2_client()
            result = storage.run_backup(manifest_path=manifest_path, root=self.root, prefix=prefix,
                                        receipt_path=receipt_path, execute=True, client=self.client)
            receipt_copy = None
            if receipt_path.exists():
                receipt_copy = self.budget.keep_receipt(receipt_path, f"{BASE}/backup-history/{key}/after-{attempt:02d}.json.txt")
            failed = retryable_backup_failure(result, manifest_sha=manifest_sha, prefix=prefix, files=files)
            if not failed or not self.budget.available(key):
                return result
            if receipt_copy is None or read(checked(self.root, receipt_copy)) != result:
                raise SupplementError("actual_byteexact_failed_backup_receipt_required_before_retry")
            if core.sha_file(manifest_path)[0] != manifest_sha:
                raise SupplementError("original_backup_manifest_changed_no_retry")
            self.budget.retry(key, ledger=self.ledger, reservation=reservation,
                              failure={"http_status": failed[0]["http_status"], "failed_entries": failed,
                                       "canonical_failed_receipt_opaque": receipt_copy})

    def close(self):
        if self.client is not None:
            self.client.close()

    def restore(self, **kwargs):
        """Original exact restoration, SDK retry off; no new restore retry loop."""
        if Path(kwargs.get("root")) != self.root or kwargs.get("execute") is not True:
            raise SupplementError("only_original_executing_restore_allowed")
        report = path(self.root, Path(kwargs["report_path"]).relative_to(self.root).as_posix())
        if not report.relative_to(self.root).as_posix().startswith(f"audit/source_extensions_20261019/Yang2025/{PLAN_SHA}/"):
            raise SupplementError("restore_report_outside_original_record_namespace")
        key = core.digest({"original_restore_report": report.relative_to(self.root).as_posix()})
        if report.exists():
            self.budget.keep_receipt(report, f"{BASE}/restore-history/{key}/before.json.txt")
        if self.client is None:
            self.client = create_r2_client()
        try:
            return storage.run_restore(**kwargs, client=self.client)
        finally:
            if report.exists():
                self.budget.keep_receipt(report, f"{BASE}/restore-history/{key}/after.json.txt")


def verify_retry_proof(root, item, ledger, *, minimum_history_index=0):
    state = read(checked(root, item))
    counts = state.get("logical_attempts", {})
    n = state.get("extra_logical_retries")
    if (state.get("schema_version") != SCHEMA or type(n) is not int or not 0 <= n <= MAX_EXTRA_RETRIES
            or state.get("active_retry") is not None or not isinstance(counts, dict)
            or any(type(v) is not int or not 1 <= v <= MAX_ATTEMPTS for v in counts.values())
            or sum(v - 1 for v in counts.values()) != n or len(state.get("retry_receipts", [])) != n):
        raise SupplementError("bounded_durable_logical_retry_proof_invalid")
    rebuilt = {key: 1 for key in counts}
    last_index = minimum_history_index - 1
    for sequence, descriptor in enumerate(state["retry_receipts"], 1):
        value = read(checked(root, descriptor));intent = read(checked(root, value["intent"]))
        index = value.get("failed_original_history_index")
        key = intent.get("logical_key")
        if key not in rebuilt:
            raise SupplementError("retry_logical_key_not_in_durable_counter")
        rebuilt[key] += 1
        if (intent.get("extra_retry_number") != sequence or intent.get("amendment") != state.get("amendment")
                or intent.get("next_logical_attempt") != rebuilt[key]
                or intent.get("failure", {}).get("http_status") not in RETRY_STATUSES
                or intent.get("backoff_seconds") not in BACKOFF or type(index) is not int
                or index <= last_index or index + 1 >= len(ledger["history"])
                or value.get("failed_transfer") != ledger["history"][index]
                or value["failed_transfer"].get("status") != "FAILURE_FULL_RESERVATION_CONSUMED"
                or value["failed_transfer"].get("verified_bytes") is not None):
            raise SupplementError("retry_failure_must_be_fully_charged_in_original_history")
        last_index = index
        before, after = ledger["history"][index], ledger["history"][index + 1]
        if (value.get("next_original_reservation") != intent.get("original_reservation")
                or any(before[k] != after[k] for k in ("kind", "reserved_bytes", "identity"))
                or any(before[k] != intent["original_reservation"][k] for k in ("kind", "reserved_bytes", "identity"))):
            raise SupplementError("retry_reservation_identity_or_size_changed")
        failed_receipt = intent.get("failure", {}).get("canonical_failed_receipt_opaque")
        if failed_receipt:
            checked(root, failed_receipt)
    if rebuilt != counts:
        raise SupplementError("logical_attempt_counters_disagree_with_immutable_retry_chain")
    snapshots = state.get("IO_receipt_snapshots")
    if not isinstance(snapshots, list):
        raise SupplementError("all_IO_receipt_snapshots_must_be_retained")
    seen = set()
    for item in snapshots:
        descriptor = item.get("opaque_snapshot", {})
        if (not descriptor.get("path", "").endswith(".txt") or descriptor["path"] in seen
                or not descriptor["path"].startswith((BASE + "/backup-history/", BASE + "/restore-history/"))):
            raise SupplementError("opaque_IO_receipt_inventory_changed")
        seen.add(descriptor["path"])
        checked(root, descriptor)
    return state


def global_aggregation_fingerprints(root):
    names = [f"audit/source_extensions_20261019/Yang2025/{PLAN_SHA}/{name}" for name in
             ("coordinator_status.json", "native_inventory.json", "unit_evidence.json", "role_assignment.json",
              "final_backup_manifest.json", "final_backup_receipt.json")]
    names.append(f"cache/source_extensions_20261019/Yang2025/{PLAN_SHA}/source_cache.json")
    return [{"source_at_snapshot": name, "sha256_at_snapshot": core.sha_file(path(root, name))[0]
             if path(root, name).exists() else None} for name in names]


def verify_completion(root):
    """Read-only proof; never locks, builds clients/Ledger or prepares records."""
    root, plan, _ = load(root)
    amendment, amendment_ref = verify_amendment(root, plan)
    filename = path(root, BASE + "/completion.json")
    value = read(filename)
    if (value.get("schema_version") != SCHEMA or value.get("status") != "ALL153_RECORDS_PREPARED_ORIGINAL_V2_FINALIZATION_REQUIRED"
            or value.get("amendment") != amendment_ref or value.get("global_finalization_performed") is not False
            or value.get("original_prepare_calls") != 0 or value.get("verified_records") != RECORD_COUNT):
        raise SupplementError("supplement_is_not_a_global_completion_proof")
    claim = read(checked(root, value["execution_claim"]))
    if claim != {"schema_version": SCHEMA, "amendment": amendment_ref, "invocations": 1}:
        raise SupplementError("single_supplement_execution_claim_required")
    before, _ = closed(root, plan, checked(root, amendment["initial_closed_ledger"]))
    after, _ = closed(root, plan, checked(root, value["after_closed_ledger_before_completion_pointer"]))
    parent._prefix(before, after)
    if after.get("explicit_range_transport_amendment") != amendment_ref:
        raise SupplementError("original_ledger_must_retain_explicit_amendment")
    retry_state = verify_retry_proof(root, value["retry_ledger"], after, minimum_history_index=len(before["history"]))
    if retry_state["amendment"] != amendment_ref:
        raise SupplementError("retry_ledger_amendment_changed")
    verified, records, _ = record_states(root, plan, require_all=True)
    if verified != value.get("verified_record_ids"):
        raise SupplementError("all153_original_record_proof_changed")
    if value.get("record_proofs") != [{"record_id": record["record_id"], "state": ref(root, filename),
            "raw_backup_receipt": state["raw_backup_receipt"], "cache_backup_receipt": state["cache_backup_receipt"],
            "artifacts": state["artifacts"]} for record, filename, state in records]:
        raise SupplementError("exact_all153_state_and_dual_GET_proof_roster_required")
    current, _ = closed(root, plan)
    parent._prefix(after, current)
    if (current.get("explicit_range_transport_amendment") != amendment_ref
            or current.get("explicit_range_transport_completion") != ref(root, filename)):
        raise SupplementError("closed_ledger_supplement_completion_pointer_missing")
    return {"status": value["status"], "completion": ref(root, filename), "verified_records": len(verified),
            "original_V2_finalization_still_required": True, "new_HTTP_or_R2_requests": 0,
            "training_started": False, "GPU_actions": 0}


def run(root, *, execute=False):
    root, plan, plan_ref = load(root)
    if not execute:
        return {"schema_version": SCHEMA, "status": "DRY_RUN_ONE_SUPPLEMENT_NO_NETWORK", "frozen_plan": plan_ref,
                "transport_policy": policy(), "amendment_path": AMENDMENT, "original_prepare_calls": 0,
                "global_finalization_allowed": False, "training_started": False, "GPU_actions": 0}
    if path(root, BASE + "/completion.json").exists():
        return verify_completion(root)
    with parent.owner(root), parent.previous.coordinator_lock(root):
        amendment, amendment_ref = verify_amendment(root, plan)
        claim_file = path(root, BASE + "/execution_claim.json")
        if claim_file.exists():
            raise SupplementError("prior_supplement_issued_no_automatic_replay")
        starting_parent(root, plan)
        verified, _, _ = record_states(root, plan)
        if verified != amendment["initial_verified_complete_record_ids"]:
            raise SupplementError("initial82_states_changed_since_authorization")
        core.atomic_json(claim_file, {"schema_version": SCHEMA, "amendment": amendment_ref, "invocations": 1}, immutable=True)
        aggregations_before = global_aggregation_fingerprints(root)
        original = read(ledger_file(root))
        if "explicit_range_transport_amendment" in original or "explicit_range_transport_completion" in original:
            raise SupplementError("existing_supplement_pointer_never_overwritten")
        core.atomic_json(ledger_file(root), {**original, "explicit_range_transport_amendment": amendment_ref})
        ledger = core.Ledger(ledger_file(root), plan)
        budget = session = backup = None
        error = None
        try:
            budget = RetryBudget(root, amendment_ref)
            session = parent.previous.session()
            backup = BackupWithRetries(root, ledger, budget)
            native_profile = core.validate_plan(root, plan)
            adapter = importlib.import_module(core.ADAPTERS["Yang2025"])
            cache_bytes = 0
            for record in plan["records"]:
                ledger.checkpoint()
                state = core.process_record(root, plan, record, native_profile=native_profile, ledger=ledger,
                    session=session, materialize=adapter.materialize, audit_native=None, backup_fn=backup,
                    download_fn=lambda item, target, **kwargs: download_with_retries(item, target, budget=budget, **kwargs),
                    restore_fn=backup.restore)
                cache_bytes += sum(item["bytes"] for item in state["artifacts"])
                if cache_bytes > plan["budgets"]["cache_bytes"]:
                    raise SupplementError("original_cumulative_cache_budget_exhausted")
            verified, records, _ = record_states(root, plan, require_all=True)
            if global_aggregation_fingerprints(root) != aggregations_before:
                raise SupplementError("supplement_must_not_write_any_global_aggregation_or_COMPLETE_status")
        except BaseException as caught:
            error = caught
        finally:
            # Even client-close exceptions must not leave the original wall intent open.
            try:
                if backup is not None:
                    backup.close()
            finally:
                try:
                    if session is not None:
                        session.close()
                finally:
                    ledger.checkpoint(close=True)
        after, _ = closed(root, plan)
        initial, _ = closed(root, plan, checked(root, amendment["initial_closed_ledger"]))
        parent._prefix(initial, after)
        if error is not None:
            after_ref = snapshot(root, ledger_file(root), BASE + "/failed_after_closed_ledger.json")
            failure = {"schema_version": SCHEMA, "status": "FAILED_SINGLE_SUPPLEMENT_CLOSED_NO_REPLAY",
                "amendment": amendment_ref, "execution_claim": ref(root, claim_file),
                "after_closed_ledger_before_failure_pointer": after_ref,
                "retry_ledger": ref(root, budget.filename) if budget is not None else None,
                "error_type": type(error).__name__, "global_finalization_performed": False,
                "original_prepare_calls": 0, "training_started": False, "GPU_actions": 0}
            filename = path(root, BASE + "/failure.json")
            core.atomic_json(filename, failure, immutable=True)
            core.atomic_json(ledger_file(root), {**after, "explicit_range_transport_failure": ref(root, filename)})
            raise error
        retry_ref = ref(root, budget.filename)
        verify_retry_proof(root, retry_ref, after, minimum_history_index=len(initial["history"]))
        after_ref = snapshot(root, ledger_file(root), BASE + "/after_closed_ledger.json")
        if ref(root, path(root, parent.HISTORY + "/recovery_ledger.json"))["sha256"] != PARENT_STATE_SHA:
            raise SupplementError("original_V2_call_counter_changed_during_supplement")
        completion = {"schema_version": SCHEMA,
            "status": "ALL153_RECORDS_PREPARED_ORIGINAL_V2_FINALIZATION_REQUIRED", "amendment": amendment_ref,
            "execution_claim": ref(root, claim_file), "after_closed_ledger_before_completion_pointer": after_ref,
            "retry_ledger": retry_ref, "verified_records": len(verified), "verified_record_ids": verified,
            "original_prepare_calls": 0, "global_finalization_performed": False,
            "global_aggregation_files_unchanged_during_supplement": True,
            "global_aggregation_fingerprints_before": aggregations_before,
            "original_V2_call06_state_opaque": amendment["V2_state_opaque"],
            "record_proofs": [{"record_id": record["record_id"], "state": ref(root, filename),
                "raw_backup_receipt": state["raw_backup_receipt"], "cache_backup_receipt": state["cache_backup_receipt"],
                "artifacts": state["artifacts"]} for record, filename, state in records],
            "transport_policy": policy(), "training_started": False, "GPU_actions": 0, "external_payload_accessed": False}
        filename = path(root, BASE + "/completion.json")
        core.atomic_json(filename, completion, immutable=True)
        # The immutable report binds the BEFORE-pointer snapshot, avoiding a hash cycle.
        core.atomic_json(ledger_file(root), {**after, "explicit_range_transport_completion": ref(root, filename)})
        return verify_completion(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--write-amendment", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--verify-completion", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = (write_amendment(args.root) if args.write_amendment else
                  verify_completion(args.root) if args.verify_completion else run(args.root, execute=args.execute))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"status": "BLOCKED_PRESERVE_CLAIM_LEDGER_AND_ALL_PROOFS",
            "error_type": type(error).__name__,
            "error_code": str(error) if isinstance(error, (SupplementError, core.PreparationError, parent.TransportRecoveryError))
                          else "local_or_transport_failure_details_suppressed",
            "training_started": False, "GPU_actions": 0, "external_payload_accessed": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
