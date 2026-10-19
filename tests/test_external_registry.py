"""Fabricated public-metadata fixtures; no real EEG, query, network, or fit."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from p3work import external_registry as gate


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_external_registry", ROOT / "tools/check_external_registry.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.fixture
def fixture(tmp_path):
    evidence = tmp_path / "audit/external_expansion_20261019/evidence/provider.txt"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("FABRICATED METADATA: nominal N=50; eleven canonical channels listed; no EEG\n")
    descriptor = {"path": evidence.relative_to(tmp_path).as_posix(),
                  "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                  "bytes": evidence.stat().st_size, "url": "https://example.org/fixture-metadata"}
    registry = {"schema_version": gate.SCHEMA, "scope": "METADATA_ONLY_QUALIFICATION",
        "original_protocol_archive": {"path": "archives/" + gate.PROTOCOL_NAME,
            "sha256": gate.PROTOCOL_SHA256, "bytes": gate.PROTOCOL_BYTES},
        "base_delivery": {"path": "/workspace/shared/" + gate.BASE_NAME,
            "sha256": gate.BASE_SHA256, "bytes": gate.BASE_BYTES},
        "input_contract_sha256": gate.INPUT_SHA256,
        "primary_hypothesis": {"dataset_id": "HSStroke2026_v6", "contrast": "P_vs_Bstar", "budget": 20},
        "signal_contract": {"channels_required": 11, "sampling_hz": 128,
            "context_seconds": 1.5, "crop_seconds": [0.25, 1.25]},
        "W7_protocol_freeze_sha256": None, "training_started": False,
        "external_payload_downloaded": False, "external_query_opened": False,
        "actual_training_or_evaluation_claims": False, "evidence_prefix": gate.EVIDENCE_ROOT,
        "all_evidence": [descriptor], "datasets": [{"dataset_id": "Fixture2026",
            "role": "candidate_locked_secondary", "cohort_reported": 50,
            "eligible_subjects": None, "actual_trials": None, "actual_bytes": None,
            "prior_exposure": {"status": "unexposed", "details": "Fixture only; not a claim about biological independence."},
            "channel_qualification": {"status": "metadata_compatible", "required": 11,
                "matched_count": 11, "matched_channels": list(gate.CHANNELS),
                "provenance_evidence_paths": [descriptor["path"]]},
            "selection_use_allowed": False, "payload_access_allowed": False,
            "evidence": [deepcopy(descriptor)], "remaining_gates": ["W7", "real raw schema", "external transaction"]}],
        "secondary_analysis": {"comparison": "P_vs_Bstar", "budget": 20,
            "unit": "participant", "pvalue_family_datasets": ["Fixture2026"], "multiplicity": "Holm"}}
    return tmp_path, registry


def check(fixture):
    root, registry = fixture
    return gate.validate_registry(registry, root)


def assert_invalid(fixture, field):
    report = check(fixture)
    assert report["registry_valid"] is False
    assert report["evaluation_ready"] is False
    assert any(field in error["field"] for error in report["errors"]), report


def test_public_metadata_pass_is_neither_eligible_sample_count_nor_evaluation_permission(fixture, monkeypatch):
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network attempted"))
    root, registry = fixture
    report = check(fixture)
    assert report["registry_valid"] and report["verified_metadata_files"] == 1
    assert report["eligible_external_subjects"] is None and report["evaluation_ready"] is False
    assert len(report["evaluation_blockers"]) == 3
    assert not (root / "archives").exists()  # Base archive descriptors need no payload reads.
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()) == [registry["all_evidence"][0]["path"]]


@pytest.mark.parametrize("claim", ["external_query_opened", "external_payload_downloaded", "training_started",
    "selection_use_allowed", "payload_access_allowed", "evaluation_ready", "raw_schema_verified", "fit_completed"])
def test_nested_permission_or_success_claim_cannot_escape_metadata_gate(fixture, claim):
    fixture[1]["datasets"][0][claim] = True
    assert_invalid(fixture, claim)


@pytest.mark.parametrize("field,value", [("eligible_subjects", 50), ("actual_trials", 200), ("actual_bytes", 99)])
def test_reported_people_are_not_promoted_to_measured_eligibility(fixture, field, value):
    fixture[1]["datasets"][0][field] = value
    assert_invalid(fixture, field)


@pytest.mark.parametrize("key,value", [("dataset_id", "Fixture2026"), ("contrast", "new_vs_selected"), ("budget", 5)])
def test_h1_cannot_migrate_to_new_cohort_or_new_contrast(fixture, key, value):
    fixture[1]["primary_hypothesis"][key] = value
    assert_invalid(fixture, "primary_hypothesis")


@pytest.mark.parametrize("key,value", [("channels_required", 10), ("sampling_hz", 256), ("context_seconds", 3.0), ("crop_seconds", [0, 1])])
def test_external_expansion_cannot_change_fixed_input(fixture, key, value):
    fixture[1]["signal_contract"][key] = value
    assert_invalid(fixture, "signal_contract")


@pytest.mark.parametrize("artifact", ["original_protocol_archive", "base_delivery"])
def test_correct_filename_cannot_mask_replaced_archive(fixture, artifact):
    fixture[1][artifact]["sha256"] = "0" * 64
    assert_invalid(fixture, artifact)


@pytest.mark.parametrize("field", ["sha256", "bytes"])
def test_mutated_or_truncated_evidence_is_detected(fixture, field):
    root, registry = fixture
    path = root / registry["all_evidence"][0]["path"]
    path.write_bytes(b"X" * (path.stat().st_size if field == "sha256" else 1))
    assert_invalid(fixture, "all_evidence")


@pytest.mark.parametrize("unsafe", ["../secret.txt", "/tmp/secret.txt", "audit/external_expansion_20261019/../secret.txt", "audit//secret.txt"])
def test_evidence_traversal_is_rejected_before_reading(fixture, unsafe, monkeypatch):
    fixture[1]["all_evidence"][0]["path"] = unsafe
    monkeypatch.setattr(gate, "read_metadata", lambda *a: pytest.fail("unsafe path read attempted"))
    assert_invalid(fixture, "all_evidence")


@pytest.mark.parametrize("directory_link", [False, True])
def test_symlink_cannot_escape_metadata_evidence_root(fixture, directory_link):
    root, registry = fixture
    evidence = root / registry["all_evidence"][0]["path"]
    content = evidence.read_bytes()
    evidence.unlink()
    outside = root.parent / (root.name + "-outside.txt")
    outside.write_bytes(content)
    if directory_link:
        evidence.parent.rmdir()
        evidence.parent.symlink_to(outside.parent, target_is_directory=True)
        registry["all_evidence"][0]["path"] = str(evidence.parent.relative_to(root) / outside.name)
    else:
        evidence.symlink_to(outside)
    assert_invalid(fixture, "all_evidence")


def test_raw_file_disguised_as_an_evidence_descriptor_is_not_opened(fixture):
    root, registry = fixture
    descriptor = registry["all_evidence"][0]
    old = root / descriptor["path"]
    new = old.with_suffix(".npz")
    old.rename(new)
    descriptor["path"] = new.relative_to(root).as_posix()
    assert_invalid(fixture, "all_evidence")


def test_duplicate_cohort_cannot_inflate_independent_records(fixture):
    fixture[1]["datasets"].append(deepcopy(fixture[1]["datasets"][0]))
    assert_invalid(fixture, "dataset_id")


@pytest.mark.parametrize("status", ["prior_project_exposed", "unknown"])
def test_prior_project_exposure_is_not_confirmatory_independence(fixture, status):
    fixture[1]["datasets"][0]["prior_exposure"]["status"] = status
    assert_invalid(fixture, "prior_exposure")


def test_already_exposed_data_can_be_registered_as_exploratory(fixture):
    registry = fixture[1]
    registry["datasets"][0]["role"] = "exposed_exploratory"
    registry["datasets"][0]["prior_exposure"]["status"] = "prior_project_exposed"
    registry["secondary_analysis"]["pvalue_family_datasets"] = []
    assert check(fixture)["registry_valid"]


def test_nominal_channels_do_not_prove_clinic11_compatibility(fixture):
    fixture[1]["datasets"][0]["channel_qualification"]["status"] = "nominal_only"
    assert_invalid(fixture, "matched_count")


def test_unqualified_montage_can_remain_a_locked_candidate_with_no_match_claim(fixture):
    channel = fixture[1]["datasets"][0]["channel_qualification"]
    channel.update(status="nominal_only", matched_count=None, matched_channels=None, provenance_evidence_paths=[])
    assert check(fixture)["registry_valid"]


def test_channel_match_cannot_cite_a_different_dataset_or_unlisted_paper(fixture):
    fixture[1]["datasets"][0]["channel_qualification"]["provenance_evidence_paths"] = ["audit/other.txt"]
    assert_invalid(fixture, "provenance_evidence_paths")


def test_secondary_family_cannot_drop_unfavourable_candidates_or_split_tests(fixture):
    registry = fixture[1]
    second = deepcopy(registry["datasets"][0]); second["dataset_id"] = "Second2026"
    registry["datasets"].append(second)
    assert_invalid(fixture, "pvalue_family_datasets")
    registry["secondary_analysis"]["pvalue_family_datasets"] = ["Fixture2026", "Second2026"]
    assert check(fixture)["registry_valid"]
    registry["secondary_analysis"]["multiplicity"] = "none"
    assert_invalid(fixture, "multiplicity")


def test_primary_cohort_cannot_reappear_as_new_secondary(fixture):
    registry = fixture[1]
    registry["datasets"][0]["dataset_id"] = "HSStroke2026_v6"
    registry["secondary_analysis"]["pvalue_family_datasets"] = ["HSStroke2026_v6"]
    assert_invalid(fixture, "role")


def test_cli_is_readonly_by_default_and_require_evaluation_ready_fails_closed(fixture, capsys):
    root, registry = fixture
    target = root / "registry.json"; target.write_text(json.dumps(registry))
    before = sorted(root.rglob("*"))
    assert cli.main(["--root", str(root), "--registry", "registry.json"]) == 0
    assert json.loads(capsys.readouterr().out)["evaluation_ready"] is False
    assert sorted(root.rglob("*")) == before
    assert cli.main(["--root", str(root), "--registry", "registry.json", "--require-evaluation-ready"]) == 1
    assert json.loads(capsys.readouterr().out)["registry_valid"] is True


def test_explicit_report_cannot_overwrite_evidence(fixture, capsys):
    root, registry = fixture
    (root / "registry.json").write_text(json.dumps(registry))
    evidence = registry["all_evidence"][0]["path"]
    original = (root / evidence).read_bytes()
    assert cli.main(["--root", str(root), "--registry", "registry.json", "--output", evidence]) == 2
    assert (root / evidence).read_bytes() == original
    assert json.loads(capsys.readouterr().out)["errors"][0]["field"] == "output"


def test_explicit_new_report_records_real_metadata_check_only(fixture, capsys):
    root, registry = fixture
    (root / "registry.json").write_text(json.dumps(registry))
    assert cli.main(["--root", str(root), "--registry", "registry.json", "--output", "audit/check.json"]) == 0
    report = json.loads((root / "audit/check.json").read_text())
    assert report["registry_valid"] is True and report["evaluation_ready"] is False
    capsys.readouterr()


@pytest.mark.parametrize("field", ["W7_protocol_freeze_sha256", "eligible_subjects"])
def test_unknown_actual_status_must_be_explicit_not_an_omitted_field(fixture, field):
    target = fixture[1] if field.startswith("W7") else fixture[1]["datasets"][0]
    del target[field]
    assert_invalid(fixture, field)


def test_invalid_evidence_prefix_cannot_cause_a_read_outside_audit_allowlist(fixture, monkeypatch):
    registry = fixture[1]
    registry["evidence_prefix"] = "data/"
    registry["all_evidence"][0]["path"] = "data/private_query.json"
    monkeypatch.setattr(gate, "read_metadata", lambda *a: pytest.fail("out-of-scope metadata read attempted"))
    assert_invalid(fixture, "evidence_prefix")


@pytest.mark.parametrize("channels", [list(gate.CHANNELS)[::-1], list(gate.CHANNELS[:-1]) + ["CPz"], ["Cz"] * 11])
def test_channel_count_alone_cannot_substitute_wrong_reordered_or_duplicate_names(fixture, channels):
    fixture[1]["datasets"][0]["channel_qualification"]["matched_channels"] = channels
    assert_invalid(fixture, "matched_channels")


def test_incompatible_montage_can_be_retained_without_changing_the_fixed_input(fixture):
    registry = fixture[1]
    dataset = registry["datasets"][0]
    dataset["role"] = "montage_sensitivity_reserve"
    dataset["channel_qualification"].update(status="incompatible", matched_count=10,
                                          matched_channels=list(gate.CHANNELS[:-1]))
    registry["secondary_analysis"]["pvalue_family_datasets"] = []
    assert check(fixture)["registry_valid"]


@pytest.mark.parametrize("dataset_id", ["Dreyer2023", "Kumar2024", "Yang2025_2C"])
def test_known_paper1_paper2_data_cannot_erase_recorded_exposure(fixture, dataset_id):
    registry = fixture[1]
    registry["datasets"][0]["dataset_id"] = dataset_id
    registry["secondary_analysis"]["pvalue_family_datasets"] = [dataset_id]
    assert_invalid(fixture, "prior_exposure")


def test_prior_project_unexposed_does_not_prove_biological_identity_independence(fixture):
    fixture[1]["datasets"][0]["biological_identity_independence_verified"] = True
    assert_invalid(fixture, "biological_identity_independence_verified")


@pytest.mark.parametrize("invalid_id", [[], {}, None, True])
def test_malformed_candidate_identity_fails_closed_instead_of_crashing(fixture, invalid_id):
    fixture[1]["datasets"][0]["dataset_id"] = invalid_id
    assert_invalid(fixture, "dataset_id")


def test_independent_namespace_addon_runs_with_no_original_package_or_archive(fixture):
    root, registry = fixture
    for relative in ("p3work/external_registry.py", "tools/check_external_registry.py"):
        target = root / relative
        target.parent.mkdir(exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    (root / "registry.json").write_text(json.dumps(registry))
    assert not (root / "p3work/__init__.py").exists()
    result = subprocess.run([sys.executable, "-B", "tools/check_external_registry.py", "--registry", "registry.json"],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["registry_valid"] is True
    result = subprocess.run([sys.executable, "-B", "tools/check_external_registry.py", "--registry", "registry.json",
                             "--require-evaluation-ready"], cwd=root, capture_output=True, text=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)["evaluation_ready"] is False


def test_non_object_json_can_emit_a_failed_explicit_report_without_crashing(fixture, capsys):
    root = fixture[0]
    (root / "registry.json").write_text("[]")
    assert cli.main(["--root", str(root), "--registry", "registry.json", "--output", "audit/bad.json"]) == 1
    assert json.loads((root / "audit/bad.json").read_text())["registry_valid"] is False
    capsys.readouterr()


def test_unknown_exposure_can_be_preregistered_without_claiming_qualification(fixture):
    dataset = fixture[1]["datasets"][0]
    dataset["prior_exposure"].update(status="unknown", details="P1/P2 artifact-wide exposure has not been audited.")
    assert_invalid(fixture, "prior_exposure")
    dataset["remaining_gates"].append("prior_project_exposure_audit")
    report = check(fixture)
    assert report["registry_valid"] is True
    assert report["evaluation_ready"] is False and report["eligible_external_subjects"] is None
    assert dataset["prior_exposure"]["status"] == "unknown"
    dataset["prior_exposure"]["status"] = "prior_project_exposed"
    assert_invalid(fixture, "prior_exposure")
