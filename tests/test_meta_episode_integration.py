"""SOFTWARE_TEST_DOUBLE: manifest-shaped CPU integration, never EEG evidence.

The reused EpisodeFixture intentionally issues synthetic status/manifest records
matching real schemas so normal gate and capability code is exercised. Its
missing internal-holdout event payloads must remain unread. No actual source
fit, scientific meta update, GPU, external payload, or download is performed.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from test_episodes import EpisodeFixture, descriptor, write

from p3work.adapt import FrozenSourceModel, module_state_sha256
from p3work.contracts import SOURCE_IDS, file_digest, validate_signal_contract
from p3work.episodes import EvaluationEpisode, EvaluationLabels
from p3work import meta_training
from p3work.meta_training import MetaJob, evaluate_development, verify_meta_inputs
from p3work.methods.lowrank import initialize_lowrank
from p3work.methods.meta_train import MetaTrainEpisode
from p3work.spatial_geometry import (
    SourceCovarianceTrial, SpatialGeometry, initialize_dictionary, source_reference,
)


SCOPE = "SOFTWARE_TEST_DOUBLE_NOT_EEG_OR_TRAINING_EVIDENCE"


def make_fixture(tmp_path, n_times=256):
    fixture = EpisodeFixture(tmp_path / SCOPE)
    write(fixture.root / "SOFTWARE_TEST_DOUBLE.json", {
        "scope": SCOPE, "synthetic": True, "research_evidence": False,
        "note": "Schema-shaped fixture complete statuses are not actual scientific completion.",
    })
    if n_times == 128:
        # A genuinely matching short-window fixture, not mislabeled 256-sample
        # inputs passed under a short-window digest.
        fixture.contract.update(selected_profile="short", context_s=[0, 2], crop_s=[.5, 1.5], n_times=128)
        fixture.contract["window_audit_receipt"]["long"]["participant_mean_coverage"] = .7
        write(fixture.contract_path, fixture.contract)
        fixture.contract_sha = validate_signal_contract(fixture.contract)
        values = fixture.values[:, :, :128].astype(np.float64)
        values -= values.mean(axis=1, keepdims=True)
        values -= values.mean(axis=2, keepdims=True)
        fixture.values = values.astype(np.float32)
        shared = fixture.root / "cache/stieger/shared_test_double_X.npy"
        np.save(shared, fixture.values[1:], allow_pickle=False)
        for session in fixture.sessions:
            session["X"] = descriptor(fixture.root, shared)
            if session["partition"] == "internal_holdout":
                continue
            rows = fixture.rows(session)
            for row in rows:
                row["context_stop_sample"] = 4000
                row["preprocessing_qc"]["filter_context_ends_at_native_sample"] = 4000
            path = fixture.root / session["events"]["path"]
            write(path, rows)
            session["events"] = descriptor(fixture.root, path)
        fixture.manifest.update(input_contract_sha256=fixture.contract_sha,
                                input_contract=descriptor(fixture.root, fixture.contract_path))
        fixture.refresh()
    return fixture


def software_source(fixture):
    with torch.random.fork_rng():
        torch.manual_seed(20261018)
        encoder = nn.Sequential(nn.Flatten(), nn.Linear(11 * fixture.contract["n_times"], 3),
                                nn.BatchNorm1d(3), nn.Tanh(), nn.Dropout(.4)).double()
        head = nn.Linear(3, 2).double()
    source = FrozenSourceModel(encoder, head, feature_dim=3)
    source.software_test_double_scope = SCOPE
    return source


def software_reference(fixture):
    return source_reference([
        SourceCovarianceTrial(dataset, "SOFTWARE_TEST_DOUBLE_SOURCE_PERSON",
                              f"{SCOPE}/{dataset}/{index}", torch.tensor(fixture.values[index + 1], dtype=torch.float64))
        for dataset in sorted(SOURCE_IDS) for index in range(2)
    ])


def parameter_for(method, *, rank=None):
    if method in {"B10", "B11"}:
        return initialize_lowrank(rank or 1, 20261018).requires_grad_(True)
    return initialize_dictionary(rank or 2, 20261018).requires_grad_(True)


def forbid_internal_payloads(monkeypatch):
    """Fail on the first attempted internal payload read, even if absent."""
    original = Path.open
    def guarded(path, *args, **kwargs):
        if "/cache/stieger/internal_holdout/" in path.as_posix():
            pytest.fail(f"{SCOPE}: attempted forbidden internal payload access")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)


@pytest.mark.parametrize("n_times", [128, 256])
@pytest.mark.parametrize("method", ["P", "B8", "B10", "B11"])
def test_validation_catalog_materializes_and_scores_all_eleven_people_without_internal_reads(
        tmp_path, monkeypatch, n_times, method):
    fixture = make_fixture(tmp_path, n_times)
    forbid_internal_payloads(monkeypatch)
    catalog = fixture.load(role="meta_validation", purpose="model_selection")
    assert set(catalog.partition_people) == set(fixture.validation)
    assert len(catalog.partition_people) == 11
    # Use the real attribute, so an accidental runner dependency on a nonexistent
    # participant_ids alias cannot be masked by an overpermissive test stub.
    assert not hasattr(catalog, "participant_ids")
    assert all(plan.query_eligible for plan in catalog.plans)
    source = software_source(fixture)
    geometry = SpatialGeometry.from_reference(software_reference(fixture).matrix)
    parameter = parameter_for(method)
    job = MetaJob(method, parameter.shape[1], .15)
    before, original_parameter = module_state_sha256(source), parameter.detach().clone()
    materialized, label_reads = [], []
    original_materialize, original_labels = catalog.materialize, EvaluationLabels.labels

    def materialize(self, plan, **kwargs):
        assert self is catalog
        episode = original_materialize(plan, **kwargs)
        assert type(episode) is EvaluationEpisode
        assert type(episode) is not MetaTrainEpisode
        assert episode.support.role == "meta_validation_support"
        assert episode.support.x.shape == (19, 11, n_times)
        assert len(episode.support.acquired_event_uids) == 20
        assert episode.prediction_inputs.x.shape == (50, 11, n_times)
        assert not hasattr(episode, "query_y")
        assert not hasattr(episode.prediction_inputs, "y")
        assert set(episode.prediction_inputs.event_uids).isdisjoint(
            episode.support.acquired_event_uids + plan.unused_boundary_run_uids)
        materialized.append(episode)
        return episode

    def labels(capability, *, purpose="model_selection", prediction_receipt=None):
        assert capability.role == "meta_validation"
        assert purpose == "model_selection"
        label_reads.append(capability.event_uids)
        return original_labels(capability, purpose=purpose, prediction_receipt=prediction_receipt)

    monkeypatch.setattr(type(catalog), "materialize", materialize)
    monkeypatch.setattr(EvaluationLabels, "labels", labels)
    result = evaluate_development(source, geometry, parameter, catalog, job)
    assert len(materialized) == len(label_reads) == len(result["sessions"]) == 11
    assert set(row["participant_id"] for row in result["participants"]) == set(fixture.validation)
    assert len(result["participants"]) == 11
    assert result["participant_mean_ba"] == pytest.approx(sum(row["ba"] for row in result["participants"]) / 11)
    assert result["participant_mean_bce"] == pytest.approx(
        sum(row["balanced_bce"] for row in result["participants"]) / 11)
    assert result["ineligible_sessions"] == []
    assert torch.equal(parameter, original_parameter) and parameter.grad is None
    assert before == module_state_sha256(source)
    assert not any(module.training for module in source.modules())
    assert all(value.grad is None for value in source.parameters())


def add_second_validation_session(fixture):
    """Reassign one fixture-only registry slot, preserving all 598 identities."""
    session = fixture.session(fixture.train[0], "2")
    person = fixture.validation[0]
    old_name = session["source_record_id"]
    session.update(participant_id=person, partition="meta_validation", source_record_id=f"{person}_Session_2.mat")
    events = fixture.root / f"cache/stieger/meta_validation/{person}/2/events.json"
    write(events, fixture.rows(session))
    session["events"] = descriptor(fixture.root, events)
    inventory_path = fixture.root / fixture.manifest["official_inventory"]["path"]
    inventory = json.loads(inventory_path.read_text())
    entry = next(item for item in inventory["files"] if item["name"] == old_name)
    entry.update(name=session["source_record_id"], provider_subject=person)
    write(inventory_path, inventory)
    fixture.manifest["official_inventory"] = descriptor(fixture.root, inventory_path)
    fixture.refresh()
    return person


def make_query_ineligible(rows):
    for row in rows[25:]:
        row.update(label=0, canonical_label="left_hand")


def test_ineligible_session_is_skipped_while_eleven_person_denominator_stays_frozen(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path)
    person = add_second_validation_session(fixture)
    fixture.mutate_rows(person, make_query_ineligible, session="1")
    forbid_internal_payloads(monkeypatch)
    catalog = fixture.load(role="meta_validation")
    ineligible = catalog.plan(person, "1")
    assert not ineligible.query_eligible and catalog.plan(person, "2").query_eligible
    original_materialize = catalog.materialize
    accessed = []
    def materialize(self, plan, **kwargs):
        assert self is catalog
        assert plan.query_eligible
        accessed.append((plan.participant_id, plan.session_id))
        return original_materialize(plan, **kwargs)
    monkeypatch.setattr(type(catalog), "materialize", materialize)
    source = software_source(fixture)
    geometry = SpatialGeometry.from_reference(software_reference(fixture).matrix)
    result = evaluate_development(source, geometry, parameter_for("P"), catalog, MetaJob("P", 2, .15))
    assert (person, "1") not in accessed and (person, "2") in accessed
    assert len(result["participants"]) == 11
    assert len(result["sessions"]) == 11
    assert result["ineligible_sessions"] == [{"participant_id": person, "session_id": "1",
                                               "reason": "predeclared_future_query_ineligible"}]
    assert next(row for row in result["participants"] if row["participant_id"] == person)["sessions"] == 1


def test_missing_evaluable_person_fails_instead_of_silently_using_ten_people(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path)
    person = fixture.validation[0]
    fixture.mutate_rows(person, make_query_ineligible)
    forbid_internal_payloads(monkeypatch)
    catalog = fixture.load(role="meta_validation")
    assert len(catalog.partition_people) == 11
    assert not catalog.plan(person, "1").query_eligible
    source = software_source(fixture)
    geometry = SpatialGeometry.from_reference(software_reference(fixture).matrix)
    with pytest.raises(ValueError, match="Every frozen validation participant"):
        evaluate_development(source, geometry, parameter_for("B8"), catalog, MetaJob("B8", 2, .15))


def mock_real_artifact_loaders(monkeypatch, fixture, *, seed=20261018):
    source, reference = software_source(fixture), software_reference(fixture)
    source_path = fixture.root / "runs/SOFTWARE_TEST_DOUBLE/source_receipt.json"
    reference_path = fixture.root / "runs/SOFTWARE_TEST_DOUBLE/reference_receipt.json"
    # Real scientific-complete receipt statuses are deliberately NOT fabricated.
    write(source_path, {"status": "SOFTWARE_TEST_DOUBLE_NOT_A_SOURCE_FIT", "scope": SCOPE})
    write(reference_path, {"status": "SOFTWARE_TEST_DOUBLE_NOT_A_SOURCE_REFERENCE", "scope": SCOPE})
    source_relative = source_path.relative_to(fixture.root).as_posix()
    reference_relative = reference_path.relative_to(fixture.root).as_posix()
    calls = []
    def load_source(root, receipt_path, contract):
        assert Path(root) == fixture.root
        assert receipt_path == source_relative and validate_signal_contract(contract) == fixture.contract_sha
        calls.append("source")
        return source, {"scope": SCOPE, "pipeline_seed": seed, "seed": seed,
                        "architecture": "SOFTWARE_TEST_DOUBLE_SMALL_LINEAR_NOT_F0"}
    def load_reference(root, receipt_path, source_receipt_path, contract):
        assert Path(root) == fixture.root
        assert receipt_path == reference_relative and source_receipt_path == source_relative
        assert validate_signal_contract(contract) == fixture.contract_sha
        calls.append("reference")
        return reference, {"scope": SCOPE}
    monkeypatch.setattr(meta_training, "load_verified_source", load_source)
    monkeypatch.setattr("p3work.source_reference_io.load_verified_reference", load_reference)
    return source, reference, source_relative, reference_relative, calls


def test_verify_meta_inputs_connects_real_catalog_interfaces_with_scoped_loader_test_doubles(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path)
    source, reference, source_receipt, reference_receipt, calls = mock_real_artifact_loaders(monkeypatch, fixture)
    forbid_internal_payloads(monkeypatch)
    result = verify_meta_inputs(fixture.root, source_receipt=source_receipt, reference_receipt=reference_receipt,
                                split_path=fixture.split_path, job=MetaJob("B10", 1, .15))
    actual_source, actual_reference, train, validation, identity = result
    assert actual_source is source and actual_reference is reference
    assert calls == ["source", "reference"]
    assert train.role == "meta_train" and validation.role == "meta_validation"
    assert len(train.partition_people) == 40 and len(validation.partition_people) == 11
    assert set(train.partition_people).isdisjoint(validation.partition_people)
    assert set(train.partition_people + validation.partition_people).isdisjoint(fixture.internal)
    assert identity["signal_contract_sha256"] == fixture.contract_sha
    assert identity["source_model_state_sha256"] == source.frozen_state_sha256
    assert identity["source_training_event_uids_sha256"] == reference.events_sha256
    assert identity["source_receipt_sha256"] == file_digest(fixture.root / source_receipt)
    assert identity["reference_receipt_sha256"] == file_digest(fixture.root / reference_receipt)
    assert identity["source_architecture"].startswith("SOFTWARE_TEST_DOUBLE")
    assert train.final_protocol_freeze_sha256 is validation.final_protocol_freeze_sha256 is None


@pytest.mark.parametrize("stage", ["W1", "W3"])
def test_verify_meta_inputs_stops_at_data_gates_before_any_loader(tmp_path, monkeypatch, stage):
    fixture = make_fixture(tmp_path)
    fixture.tasks[stage]["status"] = "SOFTWARE_TEST_DOUBLE_PENDING"
    fixture.refresh()
    monkeypatch.setattr(meta_training, "load_verified_source", lambda *args, **kwargs: pytest.fail("premature source load"))
    with pytest.raises(ValueError, match="complete W1 and W3"):
        verify_meta_inputs(fixture.root, source_receipt="missing.json", reference_receipt="missing.json",
                           split_path=fixture.split_path, job=MetaJob("P", 2, .15))


def test_verify_meta_inputs_rejects_pipeline_seed_mismatch_before_reference_or_episode_access(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path)
    _, _, source_receipt, reference_receipt, calls = mock_real_artifact_loaders(monkeypatch, fixture, seed=20261019)
    forbid_internal_payloads(monkeypatch)
    with pytest.raises(ValueError, match="matching same-seed source"):
        verify_meta_inputs(fixture.root, source_receipt=source_receipt, reference_receipt=reference_receipt,
                           split_path=fixture.split_path, job=MetaJob("P", 2, .15))
    assert calls == ["source"]


def test_evaluate_development_rejects_training_catalog_before_materialization(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path)
    catalog = fixture.load(role="meta_train")
    monkeypatch.setattr(type(catalog), "materialize", lambda *args, **kwargs: pytest.fail("wrong-role materialization"))
    with pytest.raises(PermissionError, match="meta_validation only"):
        evaluate_development(None, None, parameter_for("P"), catalog, MetaJob("P", 2, .15))
