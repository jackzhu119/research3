"""Independent metadata-only attacks on the staged Dreyer V5 eligibility gate.

No provider payload, NumPy cache, live ledger, network, R2, fit or GPU is used.
Captured TAL JSON is already sealed by the separate all-520 qualification.
"""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from p3work.contracts import digest


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "audit/data_plan_20261020/dreyer_structure_case_review"
REVIEW = CASE_DIR / "final_native_TAL_and_backup_review.json"
CANDIDATE = CASE_DIR / "native_v5_candidate/dreyer2023.py"
FROZEN = CASE_DIR / "frozen_dreyer_native_ec2f7dbc.py"
PLAN = ROOT / "audit/source_extensions_20261019/Dreyer2023/native_structure_qualification/plan.json"
PARTIAL = "sub-40_run-3"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def candidate():
    return load_module(CANDIDATE, "_independent_v5_adversarial_candidate")


def checked_json(item):
    payload = (ROOT / item["path"]).read_bytes()
    assert len(payload) == item["bytes"]
    assert hashlib.sha256(payload).hexdigest() == item["sha256"]
    return json.loads(payload)


@pytest.fixture(scope="module")
def captures(candidate):
    report_bytes = REVIEW.read_bytes()
    assert hashlib.sha256(report_bytes).hexdigest() == candidate.QUALIFICATION_REVIEW_SHA256
    report = json.loads(report_bytes)
    assert report["status"] == "ALL520_CAPTURED_NATIVE_TAL_REVIEW_COMPLETE"
    assert report["qualification_final_proof_review"]["status"] == (
        "PASS_ALL520_NATIVE_PROOFS_522_FINAL_GET_AND_CLOSED_CONSERVED_LEDGER")
    assert report["qualification_final_proof_review"]["final_full_GET_objects"] == 522
    plan = checked_json(report["qualification_plan"])
    records = {r["record_id"]: r for r in plan["records"]}
    result = {}
    for item in report["observed_record_proofs"]:
        source = records[item["record_id"]]["raw_files"][0]
        captured = checked_json(item["native"])
        native = {
            "n_times": captured["native_n_times"],
            "record_count": captured["native_record_count"],
            "annotations": [{**e, "onset": Fraction(e["onset_token"]),
                              "duration": Fraction(e["duration_token"])}
                             for e in captured["native_annotations"]],
        }
        result[item["record_id"]] = {
            "native": native, "source": source, "captured": captured, "item": item,
            "kwargs": {"source_sha256": source["sha256"],
                       "source_record_id": Path(source["path"]).name},
        }
    assert len(result) == 520
    return result


MISSING_STOP_IDS = [
    "sub-22_run-3", "sub-22_run-4", "sub-25_run-2", "sub-27_run-1",
    "sub-27_run-3", "sub-27_run-5", "sub-29_run-5", "sub-31_run-5",
    "sub-31_run-6", "sub-33_run-1", "sub-33_run-2", "sub-43_run-2",
    "sub-43_run-3", "sub-43_run-4", "sub-45_run-5", "sub-48_run-1",
    "sub-48_run-3", "sub-48_run-5", "sub-48_run-6", "sub-59_run-2",
    "sub-85_run-4", "sub-85_run-6",
]


def test_registry_has_only_the_twenty_two_qualified_original_records(candidate, captures):
    expected = {}
    for record_id in MISSING_STOP_IDS:
        item = captures[record_id]
        expected[(item["kwargs"]["source_sha256"], item["kwargs"]["source_record_id"])] = tuple(
            e["description"] for e in item["native"]["annotations"][241:])
    assert candidate.MISSING_STOP_RECORDS == expected
    assert Counter(expected.values()) == {("1010", "33281"): 13,
        ("1010",): 1, ("1010", "33281", "33284"): 8}


@pytest.mark.parametrize("record_id", MISSING_STOP_IDS)
def test_every_missing_stop_record_accepts_its_actual_forty_complete_trials(candidate, captures, record_id):
    item = captures[record_id]
    before = deepcopy(item["native"])
    rows = candidate._trials(item["native"], **item["kwargs"])
    assert len(rows) == 40
    assert Counter(r["label"] for r in rows) == {0: 20, 1: 20}
    assert [r["original_annotation_index"] for r in rows] == list(range(4, 240, 6))
    assert item["native"] == before


@pytest.mark.parametrize("record_id", MISSING_STOP_IDS)
@pytest.mark.parametrize("identity_attack", ["absent", "wrong_sha", "wrong_filename", "wrong_pair"])
def test_missing_stop_never_grants_admission_from_a_half_matching_identity(
        candidate, captures, record_id, identity_attack):
    item = captures[record_id]
    kwargs = dict(item["kwargs"])
    if identity_attack == "absent":
        kwargs = {}
    elif identity_attack == "wrong_sha":
        kwargs["source_sha256"] = "0" * 64
    elif identity_attack == "wrong_filename":
        kwargs["source_record_id"] = "sub-01_task-R1acquisition_eeg.edf"
    else:
        other = MISSING_STOP_IDS[(MISSING_STOP_IDS.index(record_id) + 1) % 22]
        kwargs["source_record_id"] = captures[other]["kwargs"]["source_record_id"]
    with pytest.raises(ValueError):
        candidate._trials(item["native"], **kwargs)


@pytest.mark.parametrize("attack", [
    "tail_reversed", "tail_duplicate", "tail_missing", "tail_extra_stop", "tail_extra_eof",
    "tail_same_time", "tail_before_final_end", "tail_outside_clock", "eof_width",
    "eof_equals_train", "eof_equals_file_end", "unknown_marker", "broken_index",
    "nonterminal_missing_end", "nonterminal_extra_control", "wrong_label", "short_context",
])
def test_registered_missing_stop_identity_cannot_excuse_malformed_tal(candidate, captures, attack):
    item = captures["sub-43_run-3"]
    native = deepcopy(item["native"])
    a = native["annotations"]
    if attack == "tail_reversed":
        a[-2], a[-3] = a[-3], a[-2]
    elif attack == "tail_duplicate":
        a.insert(-1, dict(a[-2]))
    elif attack == "tail_missing":
        del a[-2]
    elif attack == "tail_extra_stop":
        a.insert(-1, dict(a[-2], description="32770"))
    elif attack == "tail_extra_eof":
        a.append(dict(a[-1]))
    elif attack == "tail_same_time":
        a[-2]["onset"] = a[-3]["onset"]
    elif attack == "tail_before_final_end":
        a[-3]["onset"] = a[-4]["onset"]
    elif attack == "tail_outside_clock":
        a[-1]["onset"] = Fraction(native["n_times"], 512) + 1
    elif attack == "eof_width":
        a[-1]["duration"] = Fraction(1, 512)
    elif attack == "eof_equals_train":
        a[-1]["onset"] = a[-2]["onset"]
    elif attack == "eof_equals_file_end":
        a[-1]["onset"] = Fraction(native["n_times"], 512)
    elif attack == "unknown_marker":
        a[-2]["description"] = "9999"
    elif attack == "broken_index":
        a[10]["original_annotation_index"] += 1
    elif attack == "nonterminal_missing_end":
        a[6]["description"] = "781"
    elif attack == "nonterminal_extra_control":
        a.insert(7, dict(a[-3]))
    elif attack == "wrong_label":
        a[4]["description"] = "770" if a[4]["description"] == "769" else "769"
    else:
        a[6]["onset"] = a[4]["onset"] + 1
    with pytest.raises(ValueError):
        candidate._trials(native, **item["kwargs"])


def test_partial_keeps_thirty_two_genuine_trials_and_separate_unlabelled_prefix(candidate, captures):
    item = captures[PARTIAL]
    before = deepcopy(item["native"])
    rows = candidate._trials(item["native"], **item["kwargs"])
    eligibility = candidate._native_eligibility(item["native"], rows, **item["kwargs"])
    assert len(rows) == 32 and Counter(r["label"] for r in rows) == {0: 14, 1: 18}
    assert [r["original_trial_index"] for r in rows] == list(range(1, 33))
    assert rows[-1]["original_annotation_index"] == 190
    assert rows[-1]["analysis_anchor_sample"] == 344 * 512
    assert rows[-1]["task_end_sample"] == 349 * 512
    assert eligibility["native_original_trial_start_markers"] == 33
    assert eligibility["eligible_original_MI_trials"] == 32
    assert eligibility["eligible_original_class_counts"] == {"left_hand": 14, "right_hand": 18}
    assert eligibility["labels_or_trials_synthesized"] is False
    assert eligibility["class_balancing_applied"] is False
    assert eligibility["padding_applied"] is False
    assert eligibility["experiment_stop_observed"] is False
    assert eligibility["post_trial_bookkeeping_suffix"] == []
    prefix, = eligibility["terminal_incomplete_prefixes"]
    assert prefix["original_annotations"] == item["captured"]["native_annotations"][193:]
    assert all(prefix[k] is False for k in ["cue_present", "paired800_present", "canonical_label_available",
        "MI_trial_UID_available", "included_in_MI_trials", "included_in_waveform_QC_exclusions"])
    assert not any("trial_uid" in e or "label" in e for e in prefix["original_annotations"])
    assert item["native"] == before


@pytest.mark.parametrize("attack", [
    "absent_identity", "wrong_sha", "wrong_filename", "changed_clock", "changed_records",
    "changed_prefix_onset", "changed_prefix_token", "changed_prefix_duration", "wrong_prefix_code",
    "wrong_prefix_index", "prefix_has_cue", "prefix_has_paired800", "prefix_is_not_terminal",
    "balance_label", "extra_complete_trial", "missing_complete_end", "nonterminal_prefix",
    "unknown_numeric", "negative_clock", "bad_native_order", "less_than_real_end",
])
def test_partial_exception_never_generalizes_or_synthesizes(candidate, captures, attack):
    item = captures[PARTIAL]
    native = deepcopy(item["native"])
    kwargs = dict(item["kwargs"])
    a = native["annotations"]
    if attack == "absent_identity":
        kwargs = {}
    elif attack == "wrong_sha":
        kwargs["source_sha256"] = "0" * 64
    elif attack == "wrong_filename":
        kwargs["source_record_id"] = "sub-41_task-R3online_eeg.edf"
    elif attack == "changed_clock":
        native["n_times"] += 512
    elif attack == "changed_records":
        native["record_count"] += 1
    elif attack == "changed_prefix_onset":
        a[-1]["onset"] += Fraction(1, 512)
    elif attack == "changed_prefix_token":
        a[-1]["onset_token"] = "+352.0000"
    elif attack == "changed_prefix_duration":
        a[-1]["duration"] = Fraction(1, 512)
    elif attack == "wrong_prefix_code":
        a[-1]["description"] = "33282"
    elif attack == "wrong_prefix_index":
        a[-1]["original_annotation_index"] -= 1
    elif attack == "prefix_has_cue":
        a.append(dict(a[-1], description="769", original_annotation_index=195))
    elif attack == "prefix_has_paired800":
        a.append(dict(a[-1], description="800", original_annotation_index=195))
    elif attack == "prefix_is_not_terminal":
        a.append(dict(a[-1], description="1010", original_annotation_index=195))
    elif attack == "balance_label":
        a[4]["description"] = "770" if a[4]["description"] == "769" else "769"
    elif attack == "extra_complete_trial":
        a[-2:] = deepcopy(a[1:7])
    elif attack == "missing_complete_end":
        a[6]["description"] = "781"
    elif attack == "nonterminal_prefix":
        a[3]["description"] = "786"
    elif attack == "unknown_numeric":
        a[3]["description"] = "9999"
    elif attack == "negative_clock":
        a[0]["onset"] = -1
    elif attack == "bad_native_order":
        a[10]["onset"] = a[11]["onset"] + 1
    else:
        a[-2]["onset"] = a[-1]["onset"] = a[-3]["onset"]
    with pytest.raises(ValueError):
        candidate._trials(native, **kwargs)


def test_old497_trial_rows_and_old128_uid_order_are_preserved(candidate, captures):
    frozen = load_module(FROZEN, "_independent_v5_adversarial_ec2f")
    old_pass = reused = 0
    all_uids = []
    labels = Counter()
    for item in captures.values():
        new_rows = candidate._trials(item["native"], **item["kwargs"])
        labels.update(r["label"] for r in new_rows)
        subject, _, run_id = candidate._identity(Path(item["source"]["path"]))
        uids = [digest({"dataset": "Dreyer2023", "provider_subject": str(subject),
            "physical_run_id": run_id, "source_file_sha256": item["source"]["sha256"],
            "original_annotation_index": r["original_annotation_index"],
            "native_cue_onset_token": r["native_cue_onset_token"], "raw_label_code": r["raw_label_code"]})
            for r in new_rows]
        all_uids.extend(uids)
        if item["item"]["existing_V4_gate"]["status"] == "EXISTING_V4_NATIVE_STRUCTURE_GATE_PASS":
            assert new_rows == frozen._trials(item["native"])
            old_pass += 1
        if item["item"]["reused_frozen_parent_complete"]:
            assert uids == [r["trial_uid"] for r in item["captured"]["trials"]]
            reused += 1
    assert old_pass == 497 and reused == 128
    assert labels == {0: 10394, 1: 10398}
    assert len(all_uids) == len(set(all_uids)) == 20792
    assert digest(all_uids) == "ee08f2c44b2be6a05f691c837f3a8dd64bb1376fa5644942cef702e6f94ce5ea"


def test_old_terminal_eof_at_the_same_clock_as_stop_remains_legal(candidate, captures):
    item = captures["sub-01_run-2"]
    native = deepcopy(item["native"])
    assert [e["description"] for e in native["annotations"][-2:]] == ["32770", "33284"]
    native["annotations"][-1]["onset"] = native["annotations"][-2]["onset"]
    assert len(candidate._trials(native, **item["kwargs"])) == 40


@pytest.mark.parametrize("record_id", ["sub-85_run-4", "sub-85_run-6"])
def test_only_the_two_observed_source_objects_have_zero_width_eof_at_file_end(candidate, captures, record_id):
    item = captures[record_id]
    native = item["native"]
    eof = native["annotations"][-1]
    assert eof["description"] == "33284" and eof["duration"] == 0
    assert eof["onset"] == Fraction(native["n_times"], 512) == 450
    assert len(candidate._trials(native, **item["kwargs"])) == 40
    registry_expected = {
        (captures[r]["kwargs"]["source_sha256"], captures[r]["kwargs"]["source_record_id"])
        for r in ["sub-85_run-4", "sub-85_run-6"]}
    assert candidate.MISSING_STOP_EOF_AT_FILE_END_RECORDS == registry_expected


@pytest.mark.parametrize("record_id", ["sub-85_run-4", "sub-85_run-6"])
@pytest.mark.parametrize("attack", ["beyond_file_end", "nonzero_eof_width", "same_time_as_train",
                                   "mismatched_filename", "mismatched_sha"])
def test_at_end_exception_does_not_relax_other_eof_or_source_requirements(candidate, captures, record_id, attack):
    item = captures[record_id]
    native = deepcopy(item["native"])
    kwargs = dict(item["kwargs"])
    eof = native["annotations"][-1]
    if attack == "beyond_file_end":
        eof["onset"] += Fraction(1, 512)
    elif attack == "nonzero_eof_width":
        eof["duration"] = Fraction(1, 512)
    elif attack == "same_time_as_train":
        eof["onset"] = native["annotations"][-2]["onset"]
    elif attack == "mismatched_filename":
        kwargs["source_record_id"] = "sub-43_task-R3online_eeg.edf"
    else:
        kwargs["source_sha256"] = captures["sub-43_run-3"]["kwargs"]["source_sha256"]
    with pytest.raises(ValueError):
        candidate._trials(native, **kwargs)


@pytest.mark.parametrize("record_id", ["sub-43_run-3", "sub-43_run-4", "sub-45_run-5",
                                     "sub-48_run-3", "sub-48_run-5", "sub-48_run-6"])
def test_the_other_six_missing_stop_eof_records_cannot_use_the_file_end_exception(candidate, captures, record_id):
    item = captures[record_id]
    native = deepcopy(item["native"])
    assert native["annotations"][-1]["onset"] < Fraction(native["n_times"], 512)
    native["annotations"][-1]["onset"] = Fraction(native["n_times"], 512)
    with pytest.raises(ValueError):
        candidate._trials(native, **item["kwargs"])


def test_old_legacy_eof_at_file_end_remains_rejected(candidate, captures):
    item = captures["sub-01_run-2"]
    native = deepcopy(item["native"])
    native["annotations"][-1]["onset"] = Fraction(native["n_times"], 512)
    with pytest.raises(ValueError):
        candidate._trials(native, **item["kwargs"])
