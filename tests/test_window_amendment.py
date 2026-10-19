"""Explicit SOFTWARE_TEST_DOUBLE metadata, never real EEG or scientific fits.

The temporary trust roots below are monkeypatched only inside software tests;
the production API has no caller-controlled trust-root/partial-scope argument.
"""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import pytest

from p3work import window_amendment as amendment
from p3work.contracts import CHANNELS, file_digest, validate_signal_contract
from p3work.splits import _hash
from p3work.window_acceptance import (
    WindowAcceptanceError, _approved_candidate_counts, audit_duration_batch,
    verify_approved_frozen_contract, verify_approved_window_audit, verify_frozen_window,
)
from test_window_acceptance import build_complete_software_batch, row, write
from tools import stieger_metadata_batch as base


def descriptor(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path), "bytes": path.stat().st_size}


def seal(path, content, field="artifact_sha256"):
    content = deepcopy(content)
    content.pop(field, None)
    content[field] = _hash(content)
    write(path, content)
    return content


def read(path):
    return json.loads(path.read_text())


def build_software_amendment(root):
    plan, batch, split_path = build_complete_software_batch(root)
    # Fixed original identities remain; only this synthetic template's duration
    # is 1.5s, so the old 3s/2s candidates both fail and the new one passes.
    for entry in plan["files"]:
        obj = batch / "objects" / str(entry["id"])
        rows = read(obj / "trial_metadata.json")
        for rr in rows:
            rr.update(available_feedback_seconds=1.5, triallength_reported_seconds=1.5,
                      feedback_stop_exclusive_sample=5500, resultind_matlab_one_based_inclusive=5500)
        write(obj / "trial_metadata.json", rows)
        desc = descriptor(root, obj / "trial_metadata.json")
        for filename, key in (("backup_manifest.json", "files"), ("backup_receipt.json", "objects")):
            value = read(obj / filename)
            for item in value[key]:
                if item["path"] == desc["path"]:
                    item.update(desc)
                    if key == "objects":
                        item["get"].update(sha256=desc["sha256"], bytes_read=desc["bytes"],
                                            declared_content_length=desc["bytes"])
            write(obj / filename, value)
        state = read(obj / "state.json")
        state["retained_metadata_sha256"] = {
            (obj / name).relative_to(root).as_posix(): file_digest(obj / name)
            for name in ("adapter_audit.json", "trial_metadata.json", "backup_manifest.json", "backup_receipt.json")}
        write(obj / "state.json", state)
    base.finish_durations(plan, root, batch, split_path)
    legacy = audit_duration_batch(root)
    assert legacy["exact_selected"] is legacy["floating_selected"] is None
    legacy_path = root / "audit/software_original_v2_exact.json"
    write(legacy_path, legacy)
    # A full 52-file protocol-shaped software double, not the real package.
    protocol = root / "protocol"
    protocol.mkdir()
    members = []
    for number in range(52):
        path = protocol / ("EXPERIMENT_PROTOCOL.md" if number == 0 else f"software_{number}.txt")
        path.write_text(f"SOFTWARE_TEST_DOUBLE original protocol member {number}\n")
        members.append(f"{file_digest(path)}  {path.name}")
    (protocol / "PACKAGE_CONTENTS.sha256").write_text("\n".join(members) + "\n")
    proposal = root / "audit/software_proposal.json"
    clarification = root / "audit/software_clarification.md"
    write(proposal, {"SOFTWARE_TEST_DOUBLE": True, "candidate": "one fixed 1.5s candidate"})
    clarification.write_text("SOFTWARE_TEST_DOUBLE: no extra windows or threshold changes.\n")
    approval = {
        "schema_version": amendment.APPROVAL_SCHEMA, "status": "AUTHORIZED_SINGLE_FIXED_CANDIDATE_AUDIT",
        "scientific_protocol_version": "2.1", "profile": amendment.V21_PROFILE,
        "SOFTWARE_TEST_DOUBLE": True, "user_approval": {"message_text": "批准"},
        "candidate": {"context_s": [0, 1.5], "crop_s": [.25, 1.25], "sfreq_hz": 128, "n_times": 128,
            "native_minimum_sample_span": 1500, "anchor": "verified feedback onset at cue plus 2000ms",
            "post_result_padding_permitted": False},
        "thresholds": deepcopy(amendment.THRESHOLDS),
        "population": {"partitions": deepcopy(amendment.PARTITIONS), "complete_meta_train_objects": 396,
            "original_lr_trials": 792, "replacement_of_participants_or_trials_permitted": False},
        "original_v2_files_modified": False, "external_payload_authorized": False,
        "GPU_or_remote_start_authorized": False, "W7_freeze_authorized": False,
        "original_v2_exact_artifact_sha256": legacy["artifact_sha256"],
        "approved_proposal": descriptor(root, proposal), "proposal_clarification": descriptor(root, clarification),
        "original_protocol": descriptor(root, protocol / "EXPERIMENT_PROTOCOL.md"),
        "original_package_manifest": descriptor(root, protocol / "PACKAGE_CONTENTS.sha256"),
        "original_v2_exact_audit": descriptor(root, legacy_path),
        "original_v2_window_coverage": descriptor(root, batch / "window_coverage.json"),
        "window_input_session_manifest": descriptor(root, batch / "window_input_session_manifest.json"),
        "selection_plan": descriptor(root, root / "plans/stieger_meta_duration_selection.json"),
        "participant_split": descriptor(root, split_path), "base_selection_plan_sha256": legacy["plan_sha256"],
        "split_sha256": legacy["split_sha256"], "raw_metadata_manifest_sha256": legacy["raw_metadata_manifest_sha256"],
    }
    approval_path = root / amendment.APPROVAL_PATH
    approval = seal(approval_path, approval, "approval_artifact_sha256")
    people = {}
    for person in sorted({e["provider_subject"] for e in plan["files"]}):
        sessions = {str(e["session_ordinal"]): {"original": 2, "retained": 2}
                    for e in plan["files"] if e["provider_subject"] == person}
        n = len(sessions)
        people[person] = {"original_lr_trials": 2*n, "retained": 2*n,
            "classes": {str(c): {"original": n, "retained": n} for c in (0, 1)},
            "retention": {"numerator": 1, "denominator": 1},
            "absolute_class_exclusion_gap": {"numerator": 0, "denominator": 1}, "sessions": sessions}
    candidate = {"context_seconds": [0, 1.5], "participants": people,
        "participant_mean_retention": {"numerator": 1, "denominator": 1},
        "participant_mean_absolute_class_exclusion_gap": {"numerator": 0, "denominator": 1}, "passes": True}
    audit = {"schema_version": amendment.AUDIT_SCHEMA, "status": "SINGLE_APPROVED_WINDOW_PASSED",
        "SOFTWARE_TEST_DOUBLE": True, "approval": descriptor(root, approval_path),
        "original_v2_exact_artifact_sha256": legacy["artifact_sha256"],
        **{k: approval[k] for k in ("base_selection_plan_sha256", "split_sha256", "raw_metadata_manifest_sha256",
                                   "window_input_session_manifest")},
        "profile": amendment.V21_PROFILE, "context_s": [0, 1.5], "crop_s": [.25, 1.25], "sfreq_hz": 128, "n_times": 128,
        "complete_objects": 396, "participants": 40, "original_lr_trials": 792,
        "thresholds": deepcopy(amendment.THRESHOLDS), "candidate": candidate,
        "selected_profile": amendment.V21_PROFILE, "cache_authorized": False,
        "strict_no_epsilon_difference_trials": 0, "new_candidates_evaluated": 1,
        "other_unapproved_windows_evaluated": 0, "waveform_QC_used_in_denominator": False,
        "external_payload_accessed": False, "scientific_fits_completed": 0, "GPU_or_remote_jobs_started": 0}
    audit = seal(root / amendment.AUDIT_PATH, audit)
    pins = {"APPROVAL_SHA256": file_digest(approval_path), "AUDIT_SHA256": file_digest(root / amendment.AUDIT_PATH),
        "AUDIT_ARTIFACT_SHA256": audit["artifact_sha256"], "ORIGINAL_V2_ARTIFACT_SHA256": legacy["artifact_sha256"],
        "EXPECTED_LR_TRIALS": 792}
    with patch.multiple(amendment, **pins):
        proof = verify_approved_window_audit(root)
        contract = {"schema_version": amendment.V21_SCHEMA, "status": "window_frozen", "channels": list(CHANNELS),
            "sfreq_hz": 128, "selected_profile": amendment.V21_PROFILE, "context_s": [0, 1.5],
            "crop_s": [.25, 1.25], "n_times": 128, "protocol_freeze_sha256": None,
            "window_audit_receipt": amendment.build_v21_window_receipt(proof)}
        write(root / "contracts/input_contract.json", contract)
    return pins


@pytest.fixture(scope="module")
def software_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("SOFTWARE_TEST_DOUBLE_v21")
    return root, build_software_amendment(root)


@pytest.fixture
def software_batch(software_template, tmp_path, monkeypatch):
    root, pins = software_template
    shutil.copytree(root, tmp_path, dirs_exist_ok=True)
    for key, value in pins.items():
        monkeypatch.setattr(amendment, key, value)
    return tmp_path


def test_production_trust_roots_and_only_fixed_window_are_explicit():
    assert amendment.APPROVAL_SHA256 == "2a3fcfa77b4947a49a676f4f0f611ffedc9f8b19b7233bb73c2834b589285a29"
    assert amendment.AUDIT_SHA256 == "d43b1cf7063e731a82d55d6608669865389fe68d11119fb2b28ef6d8664ecbae"
    assert amendment.EXPECTED_LR_TRIALS == 59399
    assert amendment.CONTEXT_S == (0, 1.5) and amendment.CROP_S == (.25, 1.25)
    assert amendment.NATIVE_SPAN == 1500 and amendment.N_TIMES == 128


def test_legacy_wrapper_adds_only_verified_five_file_map_without_changing_audit(tmp_path):
    build_complete_software_batch(tmp_path)
    original = audit_duration_batch(tmp_path)
    result = verify_frozen_window(tmp_path)
    assert result["artifact_files_sha256"] == {original[k]["path"]: original[k]["sha256"] for k in
        ("selection_plan", "split", "inventory", "window_input_session_manifest", "window_coverage")}
    assert "artifact_files_sha256" not in original
    assert audit_duration_batch(tmp_path) == original


def test_actual_writer_shape_preflight_and_public_gate_are_read_only(software_batch, monkeypatch):
    root = software_batch
    before = {p.relative_to(root).as_posix(): file_digest(p) for p in root.rglob("*") if p.is_file()}
    original_open = Path.open
    def metadata_only(path, *args, **kwargs):
        assert path.suffix.lower() not in {".mat", ".edf", ".npy", ".npz"}
        assert all(c not in (args[0] if args else kwargs.get("mode", "r")) for c in "wax+")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", metadata_only)
    monkeypatch.setattr(base.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network forbidden"))
    proof = verify_approved_window_audit(root)
    assert proof["cache_authorized"] is False
    assert "floating_selected" not in proof and "candidates" not in proof and "floating_coverage" not in proof
    contract = read(root / "contracts/input_contract.json")
    prepublication = verify_approved_frozen_contract(root, contract)
    assert prepublication["cache_authorized"] is False and prepublication["input_contract_sha256"] == validate_signal_contract(contract)
    result = verify_frozen_window(root)
    assert result["cache_authorized"] is True and result["status"] == "EXACT_FROZEN_WINDOW_VERIFIED"
    assert result.pop("artifact_sha256") == _hash(result)
    for key in ("selection_plan", "split", "inventory", "window_input_session_manifest", "window_coverage"):
        assert result[key]["sha256"] == file_digest(root / result[key]["path"])
    assert result["window_coverage"]["path"] == amendment.AUDIT_PATH
    after = {p.relative_to(root).as_posix(): file_digest(p) for p in root.rglob("*") if p.is_file()}
    assert after == before


def test_public_w1_gate_stays_closed_during_freeze_transaction(software_batch):
    root = software_batch
    write(root / "status/tasks.json", {"W1": {"status": "WINDOW_FREEZE_IN_PROGRESS_CLOSED"}})
    contract = read(root / "contracts/input_contract.json")
    assert verify_approved_frozen_contract(root, contract)["cache_authorized"] is False
    with pytest.raises(WindowAcceptanceError, match="w1_complete"):
        verify_frozen_window(root)


@pytest.mark.parametrize("field,value", [
    ("schema_version", "paper3-signal-v2-preparation"), ("selected_profile", "short"),
    ("context_s", [0, 1.25]), ("crop_s", [.5, 1.5]), ("n_times", 256),
    ("context_s", [False, 1.5]), ("protocol_freeze_sha256", "a"*64),
])
def test_v21_cannot_masquerade_as_short_or_change_fixed_window(software_batch, field, value):
    contract = read(software_batch / "contracts/input_contract.json")
    contract[field] = value
    with pytest.raises(ValueError):
        validate_signal_contract(contract)


@pytest.mark.parametrize("change", ["approval", "semantic_digest", "changed_session", "changed_class", "mean", "profile", "denominator", "threshold", "second_candidate", "fail"])
def test_resealed_single_audit_cannot_replace_exact_metadata_or_approval(software_batch, monkeypatch, change):
    root = software_batch
    path = root / amendment.AUDIT_PATH
    audit = read(path)
    if change == "approval": audit["approval"]["sha256"] = "a"*64
    elif change == "changed_session": audit["candidate"]["participants"]["S1"]["sessions"]["1"]["retained"] = 1
    elif change == "changed_class": audit["candidate"]["participants"]["S1"]["classes"]["0"]["retained"] -= 1
    elif change == "mean": audit["candidate"]["participant_mean_retention"] = {"numerator": 9, "denominator": 10}
    elif change == "profile": audit["profile"] = "short"
    elif change == "denominator": audit["original_lr_trials"] -= 1
    elif change == "threshold": audit["thresholds"]["participant_mean_retention_min"] = {"numerator": 3, "denominator": 4}
    elif change == "second_candidate": audit["new_candidates_evaluated"] = 2
    elif change == "fail": audit["candidate"]["passes"] = False
    if change == "semantic_digest":
        audit["artifact_sha256"] = "a"*64
        write(path, audit)
    else:
        audit = seal(path, audit)
    # A software-only reissued file pin deliberately gets past the outer hash
    # check, exercising recomputation rather than testing SHA alone.
    monkeypatch.setattr(amendment, "AUDIT_SHA256", file_digest(path))
    monkeypatch.setattr(amendment, "AUDIT_ARTIFACT_SHA256", audit["artifact_sha256"])
    with pytest.raises(WindowAcceptanceError, match="single_audit"):
        verify_approved_window_audit(root)


def test_resealed_approval_cannot_authorize_a_different_window(software_batch):
    path = software_batch / amendment.APPROVAL_PATH
    approval = read(path)
    approval["candidate"]["context_s"][1] = 1.25
    seal(path, approval, "approval_artifact_sha256")
    with pytest.raises(WindowAcceptanceError, match="fixed_user_approval_sha"):
        verify_approved_window_audit(software_batch)


@pytest.mark.parametrize("change", ["protocol", "missing_session", "get_incomplete", "contract_summary", "contract_descriptor", "symlink"])
def test_underlying_artifacts_and_contract_capabilities_remain_required(software_batch, change):
    root = software_batch
    plan = read(root / "plans/stieger_meta_duration_selection.json")
    obj = root / "audit/stieger_w1/batches" / plan["plan_sha256"] / "objects" / str(plan["files"][0]["id"])
    if change == "protocol":
        (root / "protocol/software_51.txt").write_text("changed bytes\n")
    elif change == "missing_session":
        (obj / "state.json").unlink()
    elif change == "get_incomplete":
        path = obj / "backup_receipt.json"
        content = read(path); content["objects"][-1]["get"]["complete_body_read"] = False; write(path, content)
        state = read(obj / "state.json")
        state["retained_metadata_sha256"][path.relative_to(root).as_posix()] = file_digest(path)
        write(obj / "state.json", state)
    elif change == "symlink":
        path = root / amendment.APPROVAL_PATH
        path.unlink(); path.symlink_to("/must-not-read-anything")
    else:
        path = root / "contracts/input_contract.json"; content = read(path)
        receipt = content["window_audit_receipt"]
        if change == "contract_summary":
            receipt["candidate"]["participant_mean_retention"] = {"numerator": 9, "denominator": 10}
        else:
            receipt["window_input_session_manifest"]["sha256"] = "f"*64
        write(path, content)
    with pytest.raises(WindowAcceptanceError):
        verify_frozen_window(root)


@pytest.mark.parametrize("kept_left,kept_right,passes", [(80,80,True),(79,79,False),(83,93,True),(82,94,False)])
def test_fixed_counter_uses_inclusive_exact_person_and_class_thresholds(kept_left, kept_right, passes):
    rows = []
    for person in ("S1", "S2"):
        for label, kept in ((0,kept_left),(1,kept_right)):
            rows.extend(row(person, 1, "software.mat", "a"*64, label*100+i, label,
                            1.5 if i < kept else 1) for i in range(100))
    for rr in rows:
        for key in ("feedback_stop_exclusive_sample", "resultind_matlab_one_based_inclusive"):
            rr[key] = int(rr[key])
    result, difference = _approved_candidate_counts(rows)
    assert result["passes"] is passes and difference == 0
    assert result["participant_mean_retention"] == {"numerator": Fraction(kept_left+kept_right,200).numerator,
        "denominator": Fraction(kept_left+kept_right,200).denominator}


def test_1500_sample_boundary_does_not_use_padding_or_replace_reported_length():
    rows = [row("S1", 1, "software.mat", "a"*64, i, i%2, 1.5) for i in range(4)]
    for rr in rows:
        for key in ("feedback_stop_exclusive_sample", "resultind_matlab_one_based_inclusive"):
            rr[key] = int(rr[key])
    rows[0].update(feedback_stop_exclusive_sample=5499, resultind_matlab_one_based_inclusive=5499,
                   available_feedback_seconds=1.499)
    rows[1].update(triallength_reported_seconds=1.49, available_feedback_seconds=1.49)
    result, difference = _approved_candidate_counts(rows)
    assert result["participants"]["S1"]["retained"] == 2 and difference == 0
    assert result["passes"] is False


def test_new_counter_pools_sessions_then_weights_people_equally():
    rows = []
    # S1: 400 originals, 100% retained. S2: 4 originals, 50% retained.
    # The participant mean must be 3/4, not the pooled 402/404.
    for person, sessions, each_class in (("S1", ("1", "2"), 100), ("S2", ("1",), 2)):
        for session in sessions:
            for label in (0,1):
                for i in range(each_class):
                    rr = row(person, session, "software.mat", "a"*64, i, label,
                             1.5 if person == "S1" or i == 0 else 1)
                    for key in ("feedback_stop_exclusive_sample", "resultind_matlab_one_based_inclusive"):
                        rr[key] = int(rr[key])
                    rows.append(rr)
    result, _ = _approved_candidate_counts(rows)
    assert result["participant_mean_retention"] == {"numerator": 3, "denominator": 4}
    assert result["participant_mean_absolute_class_exclusion_gap"] == {"numerator": 0, "denominator": 1}
    assert result["participants"]["S1"]["classes"]["0"]["original"] == 200
    assert set(result["participants"]["S1"]["sessions"]) == {"1", "2"}
    assert result["passes"] is False
