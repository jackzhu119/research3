#!/usr/bin/env python3
"""Read-only handoff checks. Never imports ML code, opens EEG arrays, or trains.

This is a selected metadata/integrity checker, not a complete JSON Schema engine
or a scientific validation of data/algorithms. Planning-only contracts can pass;
that does not authorize execution or certify a frozen clinical experiment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

BASE = {"BNCI2014_001", "Cho2017", "Lee2019_MI", "PhysionetMI"}
CHANNELS = ["FC3", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP4"]
HEX = re.compile(r"^[0-9a-f]{64}$")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def contract_digest(contract):
    value = json.loads(json.dumps(contract))
    value["freeze"]["protocol_freeze_sha256"] = None
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def validate(contract, registry, require_frozen=False, artifact_root=None):
    errors, warnings = [], []
    verified_artifacts = 0

    def check(condition, message):
        if not condition:
            errors.append(message)

    def section(value, key):
        result = value.get(key)
        if not isinstance(result, dict):
            errors.append("Missing/object section: " + key)
            return {}
        return result

    def strings(value, name):
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            errors.append(name + " must be a string list")
            return []
        check(len(value) == len(set(value)), name + " has duplicate entries")
        return value

    if not isinstance(contract, dict) or not isinstance(registry, dict):
        return {"errors": ["Contract and registry must both be JSON objects"], "warnings": []}
    check(contract.get("schema_version") == "paper3-work-contract-v1", "Unsupported contract schema")
    status = contract.get("status")
    check(status in ("planning_only", "frozen_for_external"), "Invalid contract status")
    frozen = status == "frozen_for_external"
    check(contract.get("this_delivery_executes_eeg_or_training") is False,
          "The handoff must not claim current EEG/training execution")
    if require_frozen:
        check(frozen, "Not frozen: planning-only metadata cannot pass --require-frozen")
    if not frozen:
        warnings.append("Planning-only: no model training or external evaluation is certified")

    source = set(strings(contract.get("main_source_ids"), "main_source_ids"))
    check(source == BASE, "Main f0 must use exactly the four fixed sources; Yang is a separate enhancement")
    allocation = section(registry, "main_allocation")
    check(set(strings(allocation.get("source"), "registry.main_allocation.source")) == BASE,
          "Registry main source allocation differs from the fixed four-source contract")
    groups = {key: set(strings(allocation.get(key, []), "registry." + key)) for key in
              ("development", "locked_healthy_confirmation", "locked_clinical_confirmation")}
    all_roles = {"source": source, **groups}
    role_names = sorted(all_roles)
    for i, first in enumerate(role_names):
        for second in role_names[i + 1:]:
            check(not (all_roles[first] & all_roles[second]), f"Dataset role overlap: {first}/{second}")
    check("Stieger2021" in groups["development"], "Stieger must remain the meta-development dataset")
    check("HSStroke2026_v6" in groups["locked_clinical_confirmation"], "HS v6 must remain clinically locked")
    check("Wang2026" in groups["locked_healthy_confirmation"], "Wang must remain a locked healthy study")

    records = registry.get("datasets")
    check(isinstance(records, list), "Registry datasets must be a list")
    records = records if isinstance(records, list) else []
    by_id = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("dataset_id"), str):
            errors.append("Every dataset needs a string dataset_id")
            continue
        name = record["dataset_id"]
        check(name not in by_id, "Duplicate dataset_id: " + name)
        by_id[name] = record
        if record.get("actual_eeg_schema_audit_passed") is not True:
            for field in ("n_eligible_lr_trials_verified", "n_eligible_subjects_verified", "n_eligible_sessions_verified"):
                check(record.get(field) is None, f"{name}: unverified EEG schema cannot have measured {field}")
        for field in ("n_eligible_lr_trials_verified", "n_eligible_subjects_verified", "n_eligible_sessions_verified"):
            number = record.get(field)
            check(number is None or (type(number) is int and number >= 0), f"{name}: invalid {field}")
    for name in set.union(*all_roles.values()):
        check(name in by_id, "Allocation references missing dataset: " + name)
    for name in ("HSStroke2026_v6", "Wang2026"):
        if name in by_id:
            check(by_id[name].get("train_allowed_before_final_freeze") is not True,
                  name + " must not be enabled as pre-freeze training data")

    meta = section(contract, "meta")
    check(meta.get("dataset_id") == "Stieger2021", "Wrong meta dataset")
    check(meta.get("split_unit") == "participant", "Meta split must use participants")
    check(meta.get("roster_eligibility_independent_of_candidate_window") is True,
          "Meta roster must be fixed using eligibility independent of candidate window duration")
    check(meta.get("split_frozen_before_window_choice") is True
          and meta.get("replace_subjects_after_window_qc") is False,
          "Fix the meta split before window choice and do not replace participants after window QC")
    check(meta.get("main_support_acquired_trials") == 20, "Primary meta budget must be fixed B20")
    check(meta.get("inner_updates") == 10 and meta.get("episodes_per_outer") == 4
          and meta.get("outer_updates") == 1000, "Inner/outer budget differs from METHOD_SPEC")
    check(meta.get("search_r") == [2, 4, 8] and meta.get("search_rho") == [0.05, 0.15, 0.35]
          and meta.get("search_config_count") == 9, "Unexpected primary search grid")
    check(meta.get("nested_subject_counts") == [10, 20, 40], "Unexpected nested meta scale plan")
    subjects = meta.get("subject_lists")
    eligible = meta.get("eligible_subject_ids")
    if subjects is None or eligible is None:
        (errors if frozen else warnings).append("Concrete pre-window eligible meta subject lists are not frozen")
    else:
        check(isinstance(subjects, dict), "subject_lists must be an object")
        subjects = subjects if isinstance(subjects, dict) else {}
        eligible = set(strings(eligible, "eligible_subject_ids"))
        splits = {name: set(strings(subjects.get(name), "meta." + name))
                  for name in ("train", "validation", "internal_test")}
        check(set.union(*splits.values()) == eligible, "Meta split must cover exactly the pre-window eligible ID roster")
        for a, b in (("train", "validation"), ("train", "internal_test"), ("validation", "internal_test")):
            check(not (splits[a] & splits[b]), "Participant leakage between meta " + a + " and " + b)
        if len(eligible) == 62:
            check([len(splits[k]) for k in ("train", "validation", "internal_test")] == [40, 11, 11],
                  "62 pre-window eligible participants require the fixed 40/11/11 split")
        else:
            check(bool(meta.get("reduced_roster_amendment")), "A reduced eligible roster requires an explicit split amendment")
            check(all(len(part) > 0 for part in splits.values()), "All meta partitions must be nonempty")

    seeds = section(contract, "seeds")
    expected_seeds = {"screening": [20261018], "confirmation": [20261018, 20261019, 20261020],
                      "external": [20261018, 20261019, 20261020, 20261021, 20261022]}
    check(seeds == expected_seeds, "Pipeline seed schedule differs from the frozen planning specification")
    signal = section(contract, "signal")
    check(signal.get("channels") == CHANNELS and signal.get("sfreq_hz") == 128,
          "Main signal montage/order or sample rate differs")
    check(signal.get("reference") == "CAR_exact_clinic11" and signal.get("adapter_before_joint_trial_rms") is True,
          "Spatial update must preserve the CAR contract and precede joint trial RMS")
    check(signal.get("window_decision_role") == "meta_train", "Window choice can only use meta-train metadata")
    check(signal.get("window_participant_mean_coverage_min") == 0.80
          and signal.get("window_class_exclusion_gap_max") == 0.10, "Window gate thresholds differ")
    candidates = signal.get("profile_candidates", {})
    check(candidates == {"long": {"context_s": [0, 3], "crop_s": [0.5, 2.5], "n_times": 256},
                         "short": {"context_s": [0, 2], "crop_s": [0.5, 1.5], "n_times": 128}},
          "Candidate windows differ from EXPERIMENT_PROTOCOL")
    selected, window_receipt = signal.get("selected_profile"), signal.get("window_audit_receipt")
    if selected is None or window_receipt is None:
        (errors if frozen else warnings).append("No selected, audited common signal profile yet")
    else:
        check(selected in ("long", "short"), "Unknown selected signal profile")
        check(isinstance(window_receipt, dict), "window_audit_receipt must be an object")
        if isinstance(window_receipt, dict):
            check(window_receipt.get("decision_pool_role") == "meta_train", "Window receipt used the wrong data partition")
            acceptable = []
            for name in ("long", "short"):
                measure = window_receipt.get(name, {})
                if not isinstance(measure, dict):
                    errors.append("Window measurement must be an object: " + name)
                    continue
                coverage = measure.get("participant_mean_coverage")
                gap = measure.get("participant_mean_abs_class_exclusion_gap")
                valid = (type(coverage) in (int, float) and 0 <= coverage <= 1
                         and type(gap) in (int, float) and 0 <= gap <= 1)
                check(valid, "Invalid window coverage/gap receipt for " + name)
                if valid and coverage >= 0.80 and gap <= 0.10:
                    acceptable.append(name)
            check(bool(acceptable) and selected == acceptable[0], "Chosen window does not follow the predefined gate")

    method = section(contract, "method")
    for field, value in {"id": "GENEXP", "parameterization": "r_coefficients_not_rank_r_update",
                         "reference_fit_pool": "f0_training_only", "generator_gradient_pool": "meta_train_only",
                         "encoder_parameters_and_buffers_frozen": True, "target_query_for_fit_or_selection": False,
                         "reuse_target_state_across_sessions": False}.items():
        check(method.get(field) == value, "Invalid method contract: " + field)
    primary = section(contract, "primary")
    check(primary.get("dataset_id") == "HSStroke2026_v6" and primary.get("budget_acquired_trials") == 20,
          "Primary clinical dataset/budget differs")
    check(primary.get("analysis_unit") == "participant_mean_of_session_BA", "Primary sampling unit differs")
    check(primary.get("query_min_valid_trials") == 20 and primary.get("query_min_valid_per_class") == 5,
          "Primary query eligibility differs")
    check(primary.get("comparator_selection_pool") == "meta_validation_only", "Primary comparator cannot be selected on external outcomes")
    if frozen:
        check(bool(primary.get("query_order_evidence")), "Missing acquisition-order evidence for primary query")
        check(bool(primary.get("comparator_id")), "Strongest practical comparator has not been selected/frozen")
        check(method.get("selected_r") in meta.get("search_r", []), "Selected r is absent or outside the grid")
        check(method.get("selected_rho") in meta.get("search_rho", []), "Selected rho is absent or outside the grid")
        for name in BASE | {"Stieger2021", "HSStroke2026_v6"}:
            check(by_id.get(name, {}).get("actual_eeg_schema_audit_passed") is True,
                  "Real EEG schema gate not passed for " + name)
    secondary = section(contract, "secondary")
    check(secondary.get("fixed_query_label_access_budgets") == [0, 4, 8, 12, 16, 20], "Secondary curve must share its predefined query boundary")
    check(secondary.get("gate_required_for_primary") is False and secondary.get("mix_budget_meta_is_secondary") is True,
          "Secondary modules must not redefine the primary algorithm")
    check(secondary.get("label_access_is_not_acquisition_savings") is True, "Do not equate hidden labels with unacquired cued trials")
    reuse = section(contract, "legacy_reuse")
    check(all(reuse.get(key) is False for key in ("p1_processed_is_main_input", "p2_results_are_new_p3_evidence", "yang_required_by_main")),
          "Invalid direct reuse of P1/P2 or dependence on blocked Yang enhancement")
    check(reuse.get("yang_enhancement_requires_separate_gate_and_freeze") is True,
          "Yang enhancement must have its own gate/branch/freeze")

    freeze = section(contract, "freeze")
    artifacts = freeze.get("artifacts", [])
    check(isinstance(artifacts, list), "freeze.artifacts must be a list")
    artifacts = artifacts if isinstance(artifacts, list) else []
    paths, roles = set(), set()
    for row in artifacts:
        if not isinstance(row, dict):
            errors.append("Invalid artifact entry")
            continue
        name, sha, role = row.get("path"), row.get("sha256"), row.get("role")
        valid = isinstance(name, str) and isinstance(role, str) and isinstance(sha, str) and bool(HEX.fullmatch(sha))
        check(valid, "Artifact needs a relative path, role and lowercase SHA256")
        if not valid:
            continue
        path = Path(name)
        check(not path.is_absolute() and ".." not in path.parts, "Artifact path must stay within artifact root: " + name)
        check(name not in paths, "Duplicate artifact path: " + name)
        paths.add(name); roles.add(role)
        if artifact_root is not None:
            full = (artifact_root / path).resolve()
            if path.is_absolute() or ".." in path.parts:
                continue
            if not full.is_relative_to(artifact_root.resolve()):
                errors.append("Artifact resolves outside artifact root: " + name)
                continue
            if not full.is_file():
                errors.append("Missing artifact: " + name)
            elif file_hash(full) != sha:
                errors.append("Artifact SHA256 mismatch: " + name)
            else:
                verified_artifacts += 1
    if frozen:
        required = {"protocol", "methods", "data_registry", "split", "signal_contract", "reference",
                    "f0_bundle", "generator_bundle", "comparators", "analysis"}
        check(required <= roles, "Missing frozen artifact roles: " + ", ".join(sorted(required - roles)))
        recorded = freeze.get("protocol_freeze_sha256")
        check(isinstance(recorded, str) and bool(HEX.fullmatch(recorded)), "Missing protocol freeze SHA256")
        if isinstance(recorded, str) and HEX.fullmatch(recorded):
            check(recorded == contract_digest(contract), "Contract digest mismatch; protocol changed after freeze")
    elif freeze.get("protocol_freeze_sha256") is not None:
        warnings.append("A planning contract contains a freeze hash; this does not turn it into a completed experiment")
    if artifact_root is None:
        warnings.append("Artifact bytes were not opened; use --verify-artifacts for listed file SHA checks")
    warnings.append("Metadata checks do not establish real EEG units/order, biological independence, model correctness, or performance")
    return {"schema": "paper3-work-check-report-v1", "contract_status": status,
            "metadata_checks_passed": not errors, "execution_performed": False,
            "eeg_opened": False, "models_trained": 0, "artifacts_verified": verified_artifacts,
            "errors": errors, "warnings": warnings}


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=root / "schemas/WORK_CONTRACT.json")
    parser.add_argument("--registry", type=Path, default=root / "DATA_REGISTRY.json")
    parser.add_argument("--require-frozen", action="store_true")
    parser.add_argument("--verify-artifacts", action="store_true")
    parser.add_argument("--artifact-root", type=Path, default=root)
    parser.add_argument("--report", type=Path, help="Optional JSON report; no files are written when omitted")
    args = parser.parse_args(argv)
    try:
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        registry = json.loads(args.registry.read_text(encoding="utf-8"))
        report = validate(contract, registry, args.require_frozen,
                          args.artifact_root if args.verify_artifacts else None)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        report = {"metadata_checks_passed": False, "execution_performed": False,
                  "errors": [type(exc).__name__ + ": " + str(exc)], "warnings": []}
    output = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    print(output, end="")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    return 0 if report.get("metadata_checks_passed") else 1


if __name__ == "__main__":
    sys.exit(main())
