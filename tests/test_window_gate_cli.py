"""SOFTWARE_TEST_DOUBLE CLI proofs only; no actual W1 acceptance or data reads."""
from copy import deepcopy
import json

import pytest

from p3work.splits import _hash
from p3work.window_acceptance import WindowAcceptanceError
from p3work.window_amendment import V21_PROFILE
from tools import verify_window_gate as cli
from test_window_amendment import software_batch, software_template  # Full metadata-only SOFTWARE_TEST_DOUBLE fixtures.


def software_proof(version="2.1", *, formal=True):
    people = {"SOFTWARE_TEST_DOUBLE_A": {"original_lr_trials": 5, "retained": 4},
              "SOFTWARE_TEST_DOUBLE_B": {"original_lr_trials": 5, "retained": 4}}
    candidate = {"participants": people, "participant_mean_retention": {"numerator": 4, "denominator": 5},
                 "participant_mean_absolute_class_exclusion_gap": {"numerator": 1, "denominator": 20}, "passes": True}
    proof = {"SOFTWARE_TEST_DOUBLE": True, "schema_version": "paper3-approved-window-acceptance-v2.1",
             "status": "EXACT_FROZEN_WINDOW_VERIFIED" if formal else "EXACT_WINDOW_AUDIT_ONLY",
             "cache_authorized": formal, "participants": 2, "original_lr_trials": 10,
             "complete_objects": 2, "profile": V21_PROFILE, "selected_profile": V21_PROFILE,
             "input_contract_sha256": "a" * 64, "candidate": candidate}
    if version == "2.0":
        proof.update(schema_version="paper3-exact-window-acceptance-v1", exact_selected="short", floating_selected="short",
                     selection_agrees=True, floating_coverage={"SOFTWARE_TEST_DOUBLE": True, "selected": "short"},
                     candidates={"3.0": deepcopy(candidate), "2.0": deepcopy(candidate)})
        for key in ("candidate", "profile", "selected_profile"):
            proof.pop(key)
    proof["artifact_sha256"] = _hash(proof)
    return proof


@pytest.mark.parametrize("version", ["2.0", "2.1"])
def test_default_cli_summarizes_verified_version_without_writes(tmp_path, monkeypatch, capsys, version):
    proof = software_proof(version)
    before = deepcopy(proof)
    monkeypatch.setattr(cli, "verify_frozen_window", lambda root: proof if root == tmp_path else pytest.fail("wrong root"))
    monkeypatch.setattr(cli, "audit_duration_batch", lambda *a, **kw: pytest.fail("Default mode must use the formal gate"))
    assert cli.main(["--root", str(tmp_path)]) == 0
    value = json.loads(capsys.readouterr().out)
    assert proof == before and not list(tmp_path.iterdir())
    assert value["full_artifact_sha256"] == before["artifact_sha256"]
    assert value.pop("artifact_sha256") == _hash(value)
    if version == "2.1":
        assert value["retained_original_lr_trials"] == 8
        assert value["candidate"]["participant_mean_retention"] == before["candidate"]["participant_mean_retention"]
        assert "participants" not in value["candidate"]
        assert not set(value).intersection({"floating_coverage", "floating_coverage_sha256", "floating_selected", "candidates", "exact_selected"})
        assert value["summary_window_version"] == "2.1-approved-single-candidate"
    else:
        assert value["floating_coverage_sha256"] == _hash(before["floating_coverage"])
        assert all("participants" not in candidate for candidate in value["candidates"].values())


def test_original_duration_diagnostic_remains_v2_and_cannot_authorize_cache(tmp_path, monkeypatch, capsys):
    proof = software_proof("2.0", formal=False)
    calls = []
    def audit(root, **kwargs):
        calls.append((root, kwargs))
        return proof
    monkeypatch.setattr(cli, "audit_duration_batch", audit)
    monkeypatch.setattr(cli, "verify_frozen_window", lambda *a: pytest.fail("Diagnostic is not a frozen window gate"))
    assert cli.main(["--root", str(tmp_path), "--audit-duration-batch"]) == 2
    value = json.loads(capsys.readouterr().out)
    assert value["cache_authorized"] is False and value["schema_version"] == "paper3-exact-window-acceptance-v1"
    assert calls[0][0] == tmp_path and not list(tmp_path.iterdir())


@pytest.mark.parametrize("option", [["--split", "other.json"], ["--selection-plan", "other.json"], ["--inventory", "other.json"]])
def test_formal_cli_refuses_override_before_any_verifier(tmp_path, monkeypatch, capsys, option):
    monkeypatch.setattr(cli, "verify_frozen_window", lambda *a: pytest.fail("Noncanonical gate was called"))
    assert cli.main(["--root", str(tmp_path), *option]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["reason"] == "formal_gate_uses_canonical_frozen_evidence_only"
    assert result["cache_authorized"] is False and not list(tmp_path.iterdir())


@pytest.mark.parametrize("mutation", ["mixed_floating", "mixed_floating_hash", "mixed_candidates", "unknown_schema", "missing_people", "bad_counts"])
def test_cli_rejects_mixed_or_incomplete_amendment_summary(tmp_path, monkeypatch, capsys, mutation):
    proof = software_proof()
    if mutation == "mixed_floating": proof["floating_coverage"] = {}
    elif mutation == "mixed_floating_hash": proof["floating_coverage_sha256"] = "a" * 64
    elif mutation == "mixed_candidates": proof["candidates"] = {}
    elif mutation == "unknown_schema": proof["schema_version"] = "unknown"
    elif mutation == "missing_people": proof["candidate"].pop("participants")
    else: proof["candidate"]["participants"]["SOFTWARE_TEST_DOUBLE_A"]["retained"] = 6
    monkeypatch.setattr(cli, "verify_frozen_window", lambda root: proof)
    assert cli.main(["--root", str(tmp_path)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["reason"] == "window_evidence_invalid" and result["cache_authorized"] is False


def test_cli_preserves_fixed_gate_error_and_false_cache_authority(tmp_path, monkeypatch, capsys):
    def blocked(root):
        raise WindowAcceptanceError("SOFTWARE_TEST_DOUBLE_MISSING_AMENDMENT")
    monkeypatch.setattr(cli, "verify_frozen_window", blocked)
    assert cli.main(["--root", str(tmp_path)]) == 2
    value = json.loads(capsys.readouterr().out)
    assert value["reason"] == "SOFTWARE_TEST_DOUBLE_MISSING_AMENDMENT" and value["cache_authorized"] is False
    proof = software_proof()
    proof["cache_authorized"] = False
    monkeypatch.setattr(cli, "verify_frozen_window", lambda root: proof)
    assert cli.main(["--root", str(tmp_path)]) == 2


def test_cli_integrates_with_actual_v21_gate_on_complete_software_metadata(software_batch, capsys):
    # The full temporary approval/audit and 396-object identities are software
    # doubles. No real native waveform exists or is opened by this integration.
    root = software_batch
    before = {path.relative_to(root).as_posix(): (path.stat().st_mtime_ns, path.stat().st_size)
              for path in root.rglob("*") if path.is_file()}
    assert cli.main(["--root", str(root)]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["schema_version"] == "paper3-approved-window-acceptance-v2.1"
    assert value["selected_profile"] == V21_PROFILE and value["cache_authorized"] is True
    assert value["retained_original_lr_trials"] == value["original_lr_trials"] == 792
    assert value["complete_objects"] == 396 and value["participants"] == 40
    assert "floating_coverage_sha256" not in value and "floating_selected" not in value
    after = {path.relative_to(root).as_posix(): (path.stat().st_mtime_ns, path.stat().st_size)
             for path in root.rglob("*") if path.is_file()}
    assert before == after
