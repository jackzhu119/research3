"""SOFTWARE_TEST_DOUBLE contract structures; no real W1 or fit evidence."""
from copy import deepcopy

import pytest

from p3work.contracts import CHANNELS, digest, validate_signal_contract
from p3work.window_amendment import (
    APPROVAL_PATH, APPROVAL_SHA256, AUDIT_PATH, AUDIT_SHA256,
    AUDIT_ARTIFACT_SHA256, ORIGINAL_V2_ARTIFACT_SHA256, V21_PROFILE,
    V21_SCHEMA, build_v21_window_receipt,
)


def software_legacy_contract(profile="short"):
    return {"SOFTWARE_TEST_DOUBLE": True, "schema_version": "paper3-signal-v2",
        "status": "window_frozen", "channels": list(CHANNELS), "sfreq_hz": 128,
        "selected_profile": profile, "context_s": [0, 3] if profile == "long" else [0, 2],
        "crop_s": [.5, 2.5] if profile == "long" else [.5, 1.5],
        "n_times": 256 if profile == "long" else 128, "protocol_freeze_sha256": None,
        "window_audit_receipt": {"decision_pool_role": "meta_train", "split_sha256": "a" * 64,
            "raw_metadata_manifest_sha256": "b" * 64,
            "long": {"participant_mean_coverage": .9 if profile == "long" else .6,
                "participant_mean_abs_class_exclusion_gap": .02},
            "short": {"participant_mean_coverage": .9,
                "participant_mean_abs_class_exclusion_gap": .02}}}


def software_approved_proof():
    descriptor = lambda path, sha: {"path": path, "sha256": sha, "bytes": 10}
    proof = {"SOFTWARE_TEST_DOUBLE": True, "cache_authorized": False,
        "approval": descriptor(APPROVAL_PATH, APPROVAL_SHA256),
        "single_candidate_audit": descriptor(AUDIT_PATH, AUDIT_SHA256),
        "window_coverage": descriptor(AUDIT_PATH, AUDIT_SHA256),
        "window_input_session_manifest": descriptor("software/session_manifest.json", "c" * 64),
        "original_v2_exact_audit": descriptor("software/old_audit.json", "d" * 64),
        "original_v2_window_coverage": descriptor("software/old_coverage.json", "e" * 64),
        "base_selection_plan_sha256": "f" * 64, "split_sha256": "a" * 64,
        "raw_metadata_manifest_sha256": "b" * 64,
        "original_v2_exact_artifact_sha256": ORIGINAL_V2_ARTIFACT_SHA256,
        "single_candidate_artifact_sha256": AUDIT_ARTIFACT_SHA256,
        "complete_objects": 396, "participants": 40, "original_lr_trials": 59399,
        "candidate": {"participant_mean_retention": {"numerator": 4, "denominator": 5},
            "participant_mean_absolute_class_exclusion_gap": {"numerator": 1, "denominator": 10},
            "passes": True}, "artifact_sha256": "8" * 64}
    return proof


def software_v21_contract():
    contract = software_legacy_contract()
    contract.update(schema_version=V21_SCHEMA, selected_profile=V21_PROFILE,
        context_s=[0, 1.5], crop_s=[.25, 1.25], n_times=128,
        window_audit_receipt=build_v21_window_receipt(software_approved_proof()))
    return contract


@pytest.mark.parametrize("profile", ["long", "short"])
def test_legacy_signal_contract_retains_both_original_profiles(profile):
    contract = software_legacy_contract(profile)
    assert validate_signal_contract(contract) == digest(contract)


def test_legacy_missing_candidate_and_nonpreferred_choice_still_rejected():
    contract = software_legacy_contract()
    contract["window_audit_receipt"]["long"] = {}
    with pytest.raises(ValueError, match="Both candidate"):
        validate_signal_contract(contract)
    contract = software_legacy_contract()
    contract["window_audit_receipt"]["long"]["participant_mean_coverage"] = .9
    with pytest.raises(ValueError, match="predeclared"):
        validate_signal_contract(contract)


def test_explicit_amendment_accepts_exact_inclusive_thresholds_structure_only():
    contract = software_v21_contract()
    assert validate_signal_contract(contract) == digest(contract)
    assert contract["SOFTWARE_TEST_DOUBLE"] is True


@pytest.mark.parametrize("field,value", [
    ("schema_version", "paper3-signal-v2"), ("selected_profile", "short"),
    ("context_s", [0, 2]), ("crop_s", [.5, 1.5]), ("n_times", 192),
    ("sfreq_hz", 256), ("protocol_freeze_sha256", "0" * 64),
])
def test_partial_or_changed_amendment_identity_cannot_use_legacy_fallback(field, value):
    contract = software_v21_contract()
    contract[field] = value
    with pytest.raises(ValueError):
        validate_signal_contract(contract)


@pytest.mark.parametrize("mutation", ["approval", "audit", "semantic", "population", "retention", "gap", "role"])
def test_amendment_requires_approval_bound_population_and_exact_gates(mutation):
    contract = software_v21_contract()
    receipt = contract["window_audit_receipt"]
    if mutation == "approval": receipt["approval"]["sha256"] = "0" * 64
    elif mutation == "audit": receipt["single_candidate_audit"]["sha256"] = "0" * 64
    elif mutation == "semantic": receipt["single_candidate_artifact_sha256"] = "0" * 64
    elif mutation == "population": receipt["original_lr_trials"] = 59400
    elif mutation == "retention": receipt["candidate"]["participant_mean_retention"] = {"numerator": 799, "denominator": 1000}
    elif mutation == "gap": receipt["candidate"]["participant_mean_absolute_class_exclusion_gap"] = {"numerator": 101, "denominator": 1000}
    else: receipt["decision_pool_role"] = "meta_validation"
    with pytest.raises(ValueError):
        validate_signal_contract(contract)


def test_validator_does_not_mutate_either_contract_version():
    for contract in (software_legacy_contract(), software_v21_contract()):
        before = deepcopy(contract)
        validate_signal_contract(contract)
        assert contract == before
