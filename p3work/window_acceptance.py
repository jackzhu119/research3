"""Read-only exact acceptance of the existing, frozen W1 window evidence.

This guard does not choose or freeze a replacement window. Its only decision
statistics are Fractions of integer original-trial counts. It reads the pinned
meta-train metadata objects, never raw EEG or another partition's session.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from functools import wraps
import json
import math
from pathlib import Path
import re

from .contracts import file_digest, validate_signal_contract
from .splits import _hash
from .window import choose_window


class WindowAcceptanceError(ValueError):
    """A fixed, credential-free reason why downstream work must stay blocked."""

    def __init__(self, code, **details):
        super().__init__(code)
        self.code, self.details = code, details


def _fail(code, **details):
    raise WindowAcceptanceError(code, **details)


def _checked_errors(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except WindowAcceptanceError:
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
            _fail("window_evidence_schema_or_integrity_invalid")
    return checked


def _local(root: Path, value) -> Path:
    path = Path(value)
    if path.is_absolute():
        try:
            path = path.relative_to(root)
        except ValueError:
            _fail("window_evidence_path_outside_root")
    if not path.parts or any(p in {"..", "."} for p in path.parts):
        _fail("window_evidence_path_invalid")
    out = root
    for part in path.parts:
        out = out / part
        if out.is_symlink():
            _fail("window_evidence_symlink_forbidden")
    return out


def _root(value):
    root = Path(value).absolute()
    for component in (root, *root.parents):
        if component.is_symlink():
            _fail("window_evidence_symlink_forbidden")
    if not root.is_dir():
        _fail("window_evidence_root_missing")
    return root


def _json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError):
        _fail("window_evidence_json_missing_or_invalid")


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _fraction(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def _native_retained(row, seconds):
    """Use the audited 1000-Hz endpoint, with the native reader's length rule.

    The reported duration is independent provider evidence. Keep the existing
    reader's 1e-9 comparison tolerance for it; it cannot extend the integer
    resultind boundary by even one sample. The original time-axis consistency
    check is trusted only through the SHA/full-GET-bound native audit and
    feedback_end_verified flag: metadata do not retain the final time-axis
    coordinate, so re-deriving that coordinate would alter reader tolerances.
    No waveform is opened here.
    """
    if (row.get("dataset") != "Stieger2021" or row.get("dataset_version") != 1
            or type(row.get("label")) is not int or row["label"] not in (0, 1)
            or type(row.get("tasknumber")) is not int or row["tasknumber"] != 1):
        _fail("native_duration_dataset_task_identity_required")
    fields = ("analysis_anchor_sample", "feedback_stop_exclusive_sample",
              "resultind_matlab_one_based_inclusive", "native_trial_samples_including_padding")
    if any(type(row.get(k)) is not int for k in fields):
        _fail("native_duration_integer_sample_proof_required")
    start, stop, resultind, total = (row[k] for k in fields)
    length, available = row.get("triallength_reported_seconds"), row.get("available_feedback_seconds")
    if (not 0 <= start < stop == resultind <= total
            or type(length) not in (int, float) or not math.isfinite(length) or length < 0
            or type(available) not in (int, float) or not math.isfinite(available)
            or available != min(length, (stop - start) / 1000)):
        _fail("native_duration_endpoint_evidence_inconsistent")
    retained = stop - start >= seconds * 1000 and length + 1e-9 >= seconds
    if retained != (available + 1e-9 >= seconds):
        _fail("native_duration_and_published_count_disagree")
    return retained


@_checked_errors
def exact_window_counts(rows: list[dict], split: dict, *, expected_session_manifest: dict) -> dict:
    """Count all original pure-LR trials; equal person weights, exact thresholds.

    choose_window supplies the existing identity/completeness checks and the
    historical floating result for comparison. Its rounded rates are never
    inputs to the exact computation. Inputs need native integer endpoint proof.
    """
    try:
        floating = choose_window(rows, split, expected_session_manifest=expected_session_manifest)
    except (ValueError, KeyError, TypeError, OverflowError):
        _fail("window_original_identity_or_completeness_validation_failed")
    groups = defaultdict(lambda: {0: [], 1: []})
    for row in rows:
        groups[row["participant_id"]][row["label"]].append(row)
    candidates = {}
    for seconds in (3, 2):
        people = {}
        retention_total, gap_total = Fraction(0), Fraction(0)
        for person in sorted(groups):
            classes = groups[person]
            counts = {str(c): {"original": len(classes[c]),
                "retained": sum(_native_retained(r, seconds) for r in classes[c])} for c in (0, 1)}
            original = sum(c["original"] for c in counts.values())
            retained = sum(c["retained"] for c in counts.values())
            retention = Fraction(retained, original)
            gap = abs(Fraction(counts["0"]["retained"], counts["0"]["original"])
                      - Fraction(counts["1"]["retained"], counts["1"]["original"]))
            retention_total += retention
            gap_total += gap
            people[person] = {"original_lr_trials": original, "retained": retained, "classes": counts,
                "retention": _fraction(retention), "absolute_class_exclusion_gap": _fraction(gap)}
        retention, gap = retention_total / len(people), gap_total / len(people)
        candidates[str(float(seconds))] = {"participants": people,
            "participant_mean_retention": _fraction(retention),
            "participant_mean_absolute_class_exclusion_gap": _fraction(gap),
            "passes": retention >= Fraction(4, 5) and gap <= Fraction(1, 10)}
    exact_selected = "long" if candidates["3.0"]["passes"] else "short" if candidates["2.0"]["passes"] else None
    return {"schema_version": "paper3-exact-window-counts-v1", "exact_selected": exact_selected,
        "floating_selected": floating["selected"], "selection_agrees": exact_selected == floating["selected"],
        "original_lr_trials": len(rows), "participants": len(groups), "candidates": candidates,
        "thresholds": {"participant_mean_retention_min": _fraction(Fraction(4, 5)),
            "participant_mean_absolute_class_exclusion_gap_max": _fraction(Fraction(1, 10))},
        "floating_coverage": floating}


def _load_selection(root, selection_plan, split_path, inventory_path):
    # Importing the existing batch module makes no downloads or writes. Its
    # exact official-plan validator is reused without changing running code.
    from tools import stieger_metadata_batch as base
    inventory_path = _local(root, inventory_path)
    plan_path = _local(root, selection_plan)
    plan = _json(plan_path)
    qualification = _json(_local(root, "plans/stieger_qualification.json"))
    try:
        base.validate_plan(qualification, inventory_path, None)
    except (base.BatchError, KeyError, TypeError, ValueError, OSError):
        _fail("qualification_plan_not_exact_official_inventory")
    if split_path is None:
        split_path = f"audit/stieger_w1/batches/{qualification['plan_sha256']}/participant_split.json"
    split_path = _local(root, split_path)
    split = _json(split_path)
    if (split.get("counts") != {"meta_train": 40, "meta_validation": 11, "internal_holdout": 11}
            or {k: len(v) for k, v in split.get("partitions", {}).items()} != split.get("counts")
            or split.get("no_post_window_reassignment") is not True):
        _fail("window_frozen_40_11_11_split_required")
    try:
        base.validate_plan(plan, inventory_path, split_path)
    except (base.BatchError, KeyError, TypeError, ValueError, OSError):
        _fail("duration_plan_not_exact_official_meta_train_selection")
    if plan.get("stage") != "meta-duration" or plan.get("file_count") != 396:
        _fail("complete_official_396_meta_train_sessions_required")
    batch_dir = _local(root, f"audit/stieger_w1/batches/{plan['plan_sha256']}")
    return base, plan, plan_path, split, split_path, inventory_path, batch_dir, qualification


def _verify_state_evidence(base, root, entry, obj, state, plan_sha):
    if (state.get("status") != "COMPLETE" or state.get("file_id") != entry["id"]
            or state.get("name") != entry["name"] or state.get("plan_sha256") != plan_sha
            or state.get("schema_blocked") is not False):
        _fail("duration_object_not_verified_complete")
    expected_paths = {str((obj / name).relative_to(root)) for name in
        ("adapter_audit.json", "trial_metadata.json", "backup_manifest.json", "backup_receipt.json")}
    hashes = state.get("retained_metadata_sha256", {})
    # Restrict before verify_completed: its general-purpose reader must never
    # follow a manifest path to raw EEG or another participant's session.
    if (not isinstance(hashes, dict) or set(hashes) != expected_paths or not all(_sha(v) for v in hashes.values())
            or state.get("backup_receipt") != str((obj / "backup_receipt.json").relative_to(root))):
        _fail("duration_object_metadata_allowlist_mismatch")
    for rel in expected_paths:
        _local(root, rel)
    raw = state.get("raw_integrity", {})
    allowed_raw_paths = {f"data/stieger2021/raw/{entry['name']}",
        f"data/stieger2021/w1-batch-temp/{entry['id']}/{entry['name']}"}
    # Raw paths are compared lexically only; there is no raw stat/open/GET.
    if (state.get("raw_path") not in allowed_raw_paths or raw.get("bytes") != entry["size"]
            or not _sha(raw.get("sha256")) or raw.get("md5") != base.official_md5(entry)[0]):
        _fail("duration_object_official_size_or_digest_mismatch")
    try:
        if not base.verify_completed(state, root):
            _fail("duration_object_not_verified_complete")
    except (base.BatchError, KeyError, TypeError, ValueError, OSError):
        _fail("duration_object_metadata_or_full_get_integrity_failed")
    return hashes


def _verify_three_object_backup(root, entry, obj, state, metadata_source_obj):
    """Bind raw + both audit objects to complete GET proofs, including copies."""
    fields = ("path", "sha256", "bytes", "role", "dataset")
    expected = [{"path": state["raw_path"], "sha256": state["raw_integrity"]["sha256"],
        "bytes": entry["size"], "role": "source_development_raw", "dataset": "Stieger2021"}]
    for name in ("adapter_audit.json", "trial_metadata.json"):
        path = obj / name
        expected.append({"path": (metadata_source_obj / name).relative_to(root).as_posix(),
            "sha256": file_digest(path), "bytes": path.stat().st_size, "role": "audit", "dataset": "Stieger2021"})
    def identities(rows):
        if not isinstance(rows, list) or len(rows) != 3 or any(not isinstance(r, dict) for r in rows):
            _fail("duration_backup_exact_three_object_manifest_required")
        return [tuple(r.get(k) for k in fields) for r in rows]
    expected_set = set(identities(expected))
    manifest, receipt = _json(obj / "backup_manifest.json"), _json(obj / "backup_receipt.json")
    manifest_ids, receipt_ids = identities(manifest.get("files")), identities(receipt.get("objects"))
    if (manifest.get("schema_version") != 1 or set(manifest_ids) != expected_set or len(set(manifest_ids)) != 3
            or receipt.get("status") != "verified" or receipt.get("execute") is not True
            or set(receipt_ids) != expected_set or len(set(receipt_ids)) != 3):
        _fail("duration_backup_manifest_receipt_object_identity_mismatch")
    for item in receipt["objects"]:
        get = item.get("get", {})
        if (item.get("status") != "verified" or get.get("status") != "verified"
                or get.get("http_status") != 200 or get.get("complete_body_read") is not True
                or get.get("sha256") != item["sha256"] or get.get("bytes_read") != item["bytes"]
                or get.get("declared_content_length") != item["bytes"]):
            _fail("duration_backup_all_three_full_get_sha_bytes_required")


def _qualification_source(base, root, entry, obj, state, qualification):
    """Verify a reused first-session copy without touching other participants."""
    if not state.get("reused_qualification_object") and not state.get("reused_qualification_state"):
        return obj, None
    if (state.get("reused_qualification_object") is not True or entry not in qualification["files"]):
        _fail("duration_reused_qualification_identity_invalid")
    prior_obj = _local(root, f"audit/stieger_w1/batches/{qualification['plan_sha256']}/objects/{entry['id']}")
    prior_path = _local(root, (prior_obj / "state.json").relative_to(root))
    if state.get("reused_qualification_state") != prior_path.relative_to(root).as_posix():
        _fail("duration_reused_qualification_path_not_exact_plan_object")
    prior = _json(prior_path)
    if prior.get("reused_qualification_object") or prior.get("reused_qualification_state"):
        _fail("duration_reused_qualification_nested_provenance_forbidden")
    prior_hashes = _verify_state_evidence(base, root, entry, prior_obj, prior, qualification["plan_sha256"])
    if prior.get("raw_integrity") != state.get("raw_integrity") or prior.get("raw_path") != state.get("raw_path"):
        _fail("duration_reused_qualification_raw_provenance_mismatch")
    for name in ("adapter_audit.json", "trial_metadata.json", "backup_manifest.json", "backup_receipt.json"):
        if (state["retained_metadata_sha256"][(obj / name).relative_to(root).as_posix()]
                != prior_hashes[(prior_obj / name).relative_to(root).as_posix()]
                or (obj / name).stat().st_size != (prior_obj / name).stat().st_size):
            _fail("duration_reused_qualification_copy_sha_or_bytes_mismatch")
    _verify_three_object_backup(root, entry, prior_obj, prior, prior_obj)
    return prior_obj, {"path": prior_path.relative_to(root).as_posix(), "sha256": file_digest(prior_path),
        "qualification_plan_sha256": qualification["plan_sha256"]}


@_checked_errors
def audit_duration_batch(root, *, selection_plan="plans/stieger_meta_duration_selection.json",
                         split_path=None, inventory_path="data_catalog/stieger2021/official_inventory.json") -> dict:
    """Read-only audit, including a completed batch whose float window failed.

    All 396 official meta-train objects must be COMPLETE first. This entry point
    never authorizes caches, even if its exact and floating selections agree.
    Paths for metadata come from plan object IDs, never a supplied manifest.
    """
    root = _root(root)
    _local(root, "plans")
    base, plan, plan_path, split, split_path, inventory_path, batch_dir, qualification = _load_selection(
        root, selection_plan, split_path, inventory_path)
    states = []
    complete = 0
    for entry in plan["files"]:
        obj = _local(root, (batch_dir / "objects" / str(entry["id"])).relative_to(root))
        path = _local(root, (obj / "state.json").relative_to(root))
        state = _json(path) if path.exists() else {}
        if (state.get("status") == "COMPLETE" and state.get("file_id") == entry["id"]
                and state.get("name") == entry["name"] and state.get("plan_sha256") == plan["plan_sha256"]
                and state.get("schema_blocked") is False):
            complete += 1
        states.append((entry, obj, state))
    if complete != len(states):
        _fail("duration_metadata_incomplete_no_window_decision", complete_objects=complete, required_objects=len(states))
    rows, sessions, object_proofs = [], [], []
    for entry, obj, state in states:
        hashes = _verify_state_evidence(base, root, entry, obj, state, plan["plan_sha256"])
        raw = state.get("raw_integrity", {})
        source_obj, qualification_proof = _qualification_source(base, root, entry, obj, state, qualification)
        _verify_three_object_backup(root, entry, obj, state, source_obj)
        audit = _json(obj / "adapter_audit.json")
        rr = _json(obj / "trial_metadata.json")
        if not isinstance(rr, list) or any(not isinstance(r, dict) for r in rr):
            _fail("duration_trial_metadata_list_required")
        rr = [r for r in rr if r.get("tasknumber") == 1 and r.get("label") is not None]
        if (audit.get("dataset") != "Stieger2021" or audit.get("source_version") != 1 or audit.get("role") != "development"
                or audit.get("participant_id") != entry["provider_subject"]
                or audit.get("session_id") != str(entry["session_ordinal"])
                or audit.get("source_file_sha256") != raw["sha256"] or audit.get("source_file_bytes") != entry["size"]
                or audit.get("native_sfreq") != 1000 or audit.get("endpoint_errors") != 0
                or type(audit.get("pure_lr_trials")) is not int or audit["pure_lr_trials"] != len(rr) or not rr
                or audit.get("pure_lr_label_counts") != {str(c): sum(r.get("label") == c for r in rr) for c in (0, 1)}):
            _fail("duration_native_audit_identity_or_counts_failed")
        if any(r.get("dataset") != "Stieger2021" or r.get("dataset_version") != 1
                or r.get("participant_id") != entry["provider_subject"] or r.get("session_id") != str(entry["session_ordinal"])
                or r.get("source_record_id") != entry["name"] or r.get("source_file_sha256") != raw["sha256"] for r in rr):
            _fail("duration_native_row_dataset_or_session_identity_failed")
        sessions.append({"participant_id": entry["provider_subject"], "session_id": str(entry["session_ordinal"]),
            "source_record_id": entry["name"], "source_file_sha256": raw["sha256"], "original_lr_trials": len(rr),
            "adapter_audit_sha256": hashes[str((obj / "adapter_audit.json").relative_to(root))],
            "trial_metadata_sha256": hashes[str((obj / "trial_metadata.json").relative_to(root))]})
        rows.extend(rr)
        object_proofs.append({"id": entry["id"], "state_sha256": file_digest(obj / "state.json"),
            "qualification_provenance": qualification_proof, **hashes})
    manifest = {"dataset": "Stieger2021", "split_sha256": plan["split_sha256"],
        "inventory_sha256": plan["inventory_sha256"], "plan_sha256": plan["plan_sha256"],
        "selection": "all_official_sessions_for_frozen_meta_train", "sessions": sessions}
    manifest["manifest_sha256"] = _hash(manifest)
    manifest_path = _local(root, (batch_dir / "window_input_session_manifest.json").relative_to(root))
    coverage_path = _local(root, (batch_dir / "window_coverage.json").relative_to(root))
    # A finished float decision normally publishes both artifacts. An audit can
    # diagnose complete metadata before publication; the formal gate cannot.
    published_manifest = manifest_path.exists()
    if published_manifest and _json(manifest_path) != manifest:
        _fail("published_window_manifest_not_exact_complete_objects")
    exact = exact_window_counts(rows, split, expected_session_manifest=manifest)
    published_coverage = coverage_path.exists()
    if published_coverage and _json(coverage_path) != exact["floating_coverage"]:
        _fail("published_window_coverage_not_recomputed_original_decision")
    def descriptor(path):
        return {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}
    result = {**exact, "schema_version": "paper3-exact-window-acceptance-v1", "status": "EXACT_WINDOW_AUDIT_ONLY",
        "cache_authorized": False, "complete_objects": len(states), "metadata_complete": True,
        "plan_sha256": plan["plan_sha256"], "split_sha256": plan["split_sha256"],
        "raw_metadata_manifest_sha256": manifest["manifest_sha256"], "official_inventory_sha256": plan["inventory_sha256"],
        "selection_plan": descriptor(plan_path), "split": descriptor(split_path), "inventory": descriptor(inventory_path),
        "window_input_session_manifest": descriptor(manifest_path) if published_manifest else None,
        "window_coverage": descriptor(coverage_path) if published_coverage else None,
        "object_metadata_evidence_sha256": _hash(object_proofs)}
    result["artifact_sha256"] = _hash(result)
    return result


@_checked_errors
def verify_frozen_window(root, *, contract_path="contracts/input_contract.json") -> dict:
    """Require exact counts, original coverage and the frozen contract to agree."""
    root = _root(root)
    tasks = _json(_local(root, "status/tasks.json"))
    if tasks.get("W1", {}).get("status") != "COMPLETE":
        _fail("w1_complete_required_for_exact_window_acceptance")
    contract_file = _local(root, contract_path)
    if contract_file != root / "contracts/input_contract.json":
        _fail("canonical_window_contract_path_required")
    contract = _json(contract_file)
    try:
        contract_sha = validate_signal_contract(contract)
    except (ValueError, TypeError, KeyError):
        _fail("frozen_signal_contract_invalid")
    proof = audit_duration_batch(root)
    selected = proof["exact_selected"]
    if selected is None or selected != proof["floating_selected"] or selected != contract["selected_profile"]:
        _fail("exact_window_selection_disagrees_with_frozen_evidence", exact_selected=selected,
              floating_selected=proof["floating_selected"], frozen_selected=contract["selected_profile"])
    receipt = contract["window_audit_receipt"]
    if (receipt.get("decision_pool_role") != "meta_train"
            or any(receipt.get(k) != proof[k] for k in ("split_sha256", "raw_metadata_manifest_sha256"))
            or receipt.get("base_selection_plan_sha256") != proof["plan_sha256"]
            or any(proof[k] is None or receipt.get(k) != proof[k] for k in
                ("window_input_session_manifest", "window_coverage"))):
        _fail("frozen_window_receipt_full_evidence_identity_mismatch")
    for profile, key in (("long", "3.0"), ("short", "2.0")):
        coverage = proof["floating_coverage"]["candidates"][key]
        if receipt.get(profile) != {"participant_mean_coverage": coverage["participant_mean_retention"],
                "participant_mean_abs_class_exclusion_gap": coverage["participant_mean_absolute_class_exclusion_gap"]}:
            _fail("frozen_window_receipt_summary_mismatch")
    proof.update(status="EXACT_FROZEN_WINDOW_VERIFIED", cache_authorized=True, input_contract_sha256=contract_sha,
        input_contract_file_sha256=file_digest(contract_file))
    proof.pop("artifact_sha256")
    proof["artifact_sha256"] = _hash(proof)
    return proof
