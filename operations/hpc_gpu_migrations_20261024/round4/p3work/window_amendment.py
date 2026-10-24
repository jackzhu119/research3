"""Pure identities for the single user-approved v2.1 window amendment.

No files, network, waveforms, model state or random generators are accessed.
The approval file digest is an explicit trust root, not a caller supplied pin.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import re

V21_SCHEMA = "paper3-signal-v2.1"
V21_PROFILE = "approved_context_1p5_v21"
APPROVAL_SCHEMA = "paper3-window-amendment-approval-v2.1"
AUDIT_SCHEMA = "paper3-approved-single-window-audit-v2.1"
RECEIPT_SCHEMA = "paper3-approved-single-window-receipt-v2.1"
APPROVAL_PATH = "plans/window_amendment_v2_1_20261019.json"
APPROVAL_SHA256 = "2a3fcfa77b4947a49a676f4f0f611ffedc9f8b19b7233bb73c2834b589285a29"
AUDIT_PATH = "audit/stieger_w1/window_amendment_v2_1_20261019/single_candidate_audit.json"
AUDIT_SHA256 = "d43b1cf7063e731a82d55d6608669865389fe68d11119fb2b28ef6d8664ecbae"
AUDIT_ARTIFACT_SHA256 = "a00b13d27ec9ddec63d66dec9d6a1ee52f121632645c5d3f59fc50b85acd6eac"
ORIGINAL_V2_ARTIFACT_SHA256 = "6133022119d6497e87f3f08bc5373c4714a14bc7f4a99969a0c9ff716b6ec8e9"
CONTEXT_S = (0, 1.5)
CROP_S = (0.25, 1.25)
SFREQ_HZ = 128
N_TIMES = 128
NATIVE_SPAN = 1500
EXPECTED_OBJECTS = 396
EXPECTED_PARTICIPANTS = 40
EXPECTED_LR_TRIALS = 59399
PARTITIONS = {"meta_train": 40, "meta_validation": 11, "internal_holdout": 11}
THRESHOLDS = {
    "participant_mean_retention_min": {"numerator": 4, "denominator": 5},
    "participant_mean_absolute_class_exclusion_gap_max": {"numerator": 1, "denominator": 10},
}


def is_v21_contract(contract):
    """Route partial/ambiguous amendment identities to its strict validator."""
    return (contract.get("schema_version") == V21_SCHEMA
            or contract.get("selected_profile") == V21_PROFILE)


def fraction_value(value):
    if (not isinstance(value, dict) or set(value) != {"numerator", "denominator"}
            or type(value["numerator"]) is not int or type(value["denominator"]) is not int
            or value["denominator"] <= 0 or value["numerator"] < 0):
        raise ValueError("Exact nonnegative integer-ratio window evidence required")
    result = Fraction(value["numerator"], value["denominator"])
    if (result.numerator, result.denominator) != (value["numerator"], value["denominator"]):
        raise ValueError("Canonical reduced window fractions required")
    return result


def _descriptor(value, *, path=None, sha=None):
    if (not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes"}
            or not isinstance(value.get("path"), str) or not value["path"]
            or value["path"].startswith("/") or any(x in {".", "..", ""} for x in value["path"].split("/"))
            or re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", ""))) is None
            or type(value.get("bytes")) is not int or value["bytes"] <= 0
            or (path is not None and value["path"] != path)
            or (sha is not None and value["sha256"] != sha)):
        raise ValueError("Pinned window amendment artifact descriptor required")


def build_v21_window_receipt(proof):
    """Build a small contract receipt from a fully verified read-only proof.

    This pure builder grants no filesystem capability. Formal callers must use
    verify_approved_window_audit and the final verify_frozen_window gate.
    """
    candidate = proof["candidate"]
    return {
        "schema_version": RECEIPT_SCHEMA, "decision_pool_role": "meta_train", "profile": V21_PROFILE,
        **{name: deepcopy(proof[name]) for name in (
            "approval", "single_candidate_audit", "window_coverage", "window_input_session_manifest",
            "original_v2_exact_audit", "original_v2_window_coverage")},
        **{name: proof[name] for name in ("base_selection_plan_sha256", "split_sha256",
            "raw_metadata_manifest_sha256", "original_v2_exact_artifact_sha256",
            "single_candidate_artifact_sha256", "complete_objects", "participants", "original_lr_trials")},
        "thresholds": deepcopy(THRESHOLDS),
        "candidate": {name: deepcopy(candidate[name]) for name in (
            "participant_mean_retention", "participant_mean_absolute_class_exclusion_gap", "passes")},
    }


def validate_v21_contract(contract):
    """Validate the amendment's immutable structure; no I/O or W7 claim."""
    if (contract.get("schema_version") != V21_SCHEMA or contract.get("selected_profile") != V21_PROFILE
            or contract.get("status") != "window_frozen"
            or contract.get("context_s") != list(CONTEXT_S) or contract.get("crop_s") != list(CROP_S)
            or any(type(v) not in {int, float} for k in ("context_s", "crop_s") for v in contract.get(k, []))
            or type(contract.get("sfreq_hz")) is not int or contract["sfreq_hz"] != SFREQ_HZ
            or type(contract.get("n_times")) is not int or contract["n_times"] != N_TIMES
            or contract.get("protocol_freeze_sha256") is not None):
        raise ValueError("Explicit approved v2.1 signal window required; W7 remains separate")
    receipt = contract.get("window_audit_receipt")
    if (not isinstance(receipt, dict) or receipt.get("schema_version") != RECEIPT_SCHEMA
            or receipt.get("decision_pool_role") != "meta_train" or receipt.get("profile") != V21_PROFILE
            or receipt.get("thresholds") != THRESHOLDS
            or receipt.get("original_v2_exact_artifact_sha256") != ORIGINAL_V2_ARTIFACT_SHA256
            or receipt.get("single_candidate_artifact_sha256") != AUDIT_ARTIFACT_SHA256
            or any(type(receipt.get(k)) is not int or receipt[k] != value for k, value in (
                ("complete_objects", EXPECTED_OBJECTS), ("participants", EXPECTED_PARTICIPANTS),
                ("original_lr_trials", EXPECTED_LR_TRIALS)))):
        raise ValueError("Approved v2.1 window evidence identity required")
    _descriptor(receipt.get("approval"), path=APPROVAL_PATH, sha=APPROVAL_SHA256)
    for name in ("single_candidate_audit", "window_coverage"):
        _descriptor(receipt.get(name), path=AUDIT_PATH, sha=AUDIT_SHA256)
    for name in ("window_input_session_manifest", "original_v2_exact_audit", "original_v2_window_coverage"):
        _descriptor(receipt.get(name))
    for name in ("base_selection_plan_sha256", "split_sha256", "raw_metadata_manifest_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(receipt.get(name, ""))) is None:
            raise ValueError("Concrete approval-bound population SHA256 required")
    candidate = receipt.get("candidate", {})
    if (not isinstance(candidate, dict)
            or set(candidate) != {"participant_mean_retention", "participant_mean_absolute_class_exclusion_gap", "passes"}
            or candidate.get("passes") is not True
            or not Fraction(4, 5) <= fraction_value(candidate["participant_mean_retention"]) <= 1
            or not 0 <= fraction_value(candidate["participant_mean_absolute_class_exclusion_gap"]) <= Fraction(1, 10)):
        raise ValueError("Unchanged exact v2.1 retention and class-gap gates required")
