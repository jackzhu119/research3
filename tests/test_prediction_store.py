"""Synthetic CPU/file fixtures only: no real prediction, fit, GPU, or network."""
import copy
from dataclasses import replace
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from p3work.adapt import PredictionResult
from p3work.contracts import SOURCE_IDS, digest, file_digest
from p3work.episodes import EvaluationLabels
from p3work import prediction_store as store
from tests.test_episodes import EpisodeFixture, descriptor, write


torch.set_num_threads(1)


def fixture(tmp_path, *, internal=False):
    native = EpisodeFixture(tmp_path)
    catalog = native.load(role="meta_validation")
    plan = catalog.plans[0]
    if internal:
        # Official-shaped metadata double for an authorized internal session.
        # Only the plan changes; its signal/label payload is never materialized.
        target = native.session(native.internal[0])
        old_prefix = f"Stieger2021:v1:{plan.participant_id}:session{plan.session_id}:"
        new_prefix = f"Stieger2021:v1:{target['participant_id']}:session{target['session_id']}:"
        def mapped(value):
            if isinstance(value, str):
                return new_prefix+value[len(old_prefix):] if value.startswith(old_prefix) else value
            if isinstance(value, list):
                return [mapped(v) for v in value]
            if isinstance(value, dict):
                return {k: mapped(v) for k, v in value.items()}
            return value
        audit = mapped(json.loads(plan.temporal_audit_json))
        audit.update(participant_id=target["participant_id"], session_id=target["session_id"], role="internal_holdout",
            purpose="internal_development_check", source_record_id=target["source_record_id"],
            source_file_sha256=target["source_file_sha256"], events_sha256=target["events"]["sha256"])
        uid_fields = ("acquired_event_uids", "support_event_uids", "query_event_uids", "unused_boundary_run_uids",
                      "incomplete_future_run_uids", "query_excluded_uids")
        plan = replace(plan, participant_id=target["participant_id"], session_id=target["session_id"], role="internal_holdout",
            temporal_audit_json=json.dumps(audit, sort_keys=True, separators=(",", ":")), temporal_audit_sha256=digest(audit),
            **{key: tuple(mapped(list(getattr(plan, key)))) for key in uid_fields})
    code = tmp_path / "p3work/methods/prediction_fixture.py"
    code.parent.mkdir(parents=True, exist_ok=True)
    code.write_text("# synthetic store fixture only\n")
    model, config, source = [tmp_path / "runs/prediction_fixture" / name for name in
                             ("model.json", "configuration.json", "source_f0_receipt.json")]
    write(model, {"test_double_not_research_evidence": True})
    write(config, {"method": "P", "rank": 2, "radius": .15, "test_double_not_research_evidence": True})
    write(source, {"status": "SOURCE_F0_COMPLETE", "synthetic": False, "training_dataset_ids": sorted(SOURCE_IDS),
                   "signal_contract_sha256": native.contract_sha, "pipeline_seed": 20261018,
                   "test_double_not_research_evidence": True})
    references = {"source_f0": descriptor(tmp_path, source), "model": descriptor(tmp_path, model),
                  "method_code": descriptor(tmp_path, code), "configuration": descriptor(tmp_path, config),
                  "input_contract": descriptor(tmp_path, native.contract_path),
                  "episode_manifest": descriptor(tmp_path, native.manifest_path),
                  "participant_split": descriptor(tmp_path, native.split_path)}
    values = torch.linspace(-1, 1, len(plan.query_event_uids), dtype=torch.float64)
    logits = torch.stack((-values, values), 1)
    logits[0] = 0  # Exact tie must predict class zero.
    result = PredictionResult(logits, logits.softmax(1), logits.argmax(1), plan.query_event_uids, native.contract_sha)
    cost = {"acquired_calibration_trials": 20, "valid_support_trials": len(plan.support_event_uids),
            "accessible_labels": len(plan.support_event_uids), "label_access_event_uids": list(plan.support_event_uids),
            "optimizer_updates": 10, "calibration_wall_seconds": .125, "gpu_compute_seconds": 0.,
            "acquisition_elapsed_seconds": None, "unused_acquired_trials": 1, "audit_trials": 0}
    access = None
    if internal:
        path = tmp_path / "manifests/internal_development_check_access.json"
        write(path, {"schema_version": "paper3-internal-check-access-v1", "status": "AUTHORIZED_ONCE_DEVELOPMENT_CHECK",
            "role": "internal_holdout", "split_sha256": plan.split_sha256,
            "input_contract_sha256": plan.input_contract_sha256, "labels_after_predictions_only": True,
            "training_and_selection_allowed": False, "selected_configuration_sha256": references["configuration"]["sha256"]})
        access = descriptor(tmp_path, path)
        native.tasks["W3"]["internal_access_receipt"] = access
        write(tmp_path / "status/tasks.json", native.tasks)
    return native, plan, result, references, cost, access


def commit(root, plan, result, references, cost, *, output="predictions/development_case", **kwargs):
    return store.commit_predictions(root, output, result, plan, method="P", pipeline_seed=20261018,
                                    references=references, cost=cost, **kwargs)


def test_commit_nonpickle_arrays_uid_and_three_cost_ledgers_and_exact_r2_manifest(tmp_path):
    _, plan, result, refs, cost, _ = fixture(tmp_path)
    with patch.object(EvaluationLabels, "labels", side_effect=AssertionError("store may not reveal labels")):
        saved = commit(tmp_path, plan, result, refs, cost)
    receipt = store.verify_prediction_receipt(tmp_path, saved["receipt"])
    assert receipt["status"] == "PREDICTIONS_COMMITTED"
    assert receipt["episode_plan_sha256"] == plan.temporal_audit_sha256
    assert receipt["labels_accessed_before_predictions"] is False and receipt["query_labels_in_store"] is False
    assert receipt["query_event_uids"] == list(plan.query_event_uids)
    output = tmp_path / "predictions/development_case"
    assert np.load(output / "predictions.npy", allow_pickle=False)[0] == 0
    assert np.array_equal(np.load(output / "logits.npy", allow_pickle=False), result.logits.numpy())
    assert np.array_equal(np.load(output / "probabilities.npy", allow_pickle=False), result.probabilities.numpy())
    recorded = json.loads((output / "cost.json").read_text())
    assert recorded["acquired_calibration_trials"] == 20
    assert recorded["valid_support_trials"] == recorded["accessible_labels"] == 19
    assert recorded["optimizer_updates"] == 10 and recorded["acquisition_elapsed_seconds"] is None
    manifest = json.loads((tmp_path / saved["artifact_manifest"]["path"]).read_text())
    assert manifest["schema_version"] == 1 and len(manifest["files"]) == 8
    assert any(row["path"].endswith("/receipt.json") for row in manifest["files"])
    assert all(row["dataset"] == "Stieger2021" and row["role"] == "result" for row in manifest["files"])
    for row in manifest["files"]:
        path = tmp_path / row["path"]
        assert path.stat().st_size == row["bytes"] and file_digest(path) == row["sha256"]
    assert "EvaluationLabels" not in inspect.signature(store.commit_predictions).parameters


def test_identical_commit_is_idempotent_and_different_prediction_or_cost_cannot_overwrite(tmp_path):
    _, plan, result, refs, cost, _ = fixture(tmp_path)
    first = commit(tmp_path, plan, result, refs, cost)
    before = {p.name: p.read_bytes() for p in (tmp_path / "predictions/development_case").iterdir() if p.is_file()}
    again = commit(tmp_path, plan, result, refs, cost)
    assert again["idempotent"] is True and again["receipt"] == first["receipt"]
    new_logits = result.logits + torch.tensor([0.5, -0.5], dtype=torch.float64)
    different = replace(result, logits=new_logits, probabilities=new_logits.softmax(1), labels=new_logits.argmax(1))
    for changed_result, changed_cost in ((different, cost), (result, dict(cost, calibration_wall_seconds=.25))):
        with pytest.raises(store.PredictionStoreError, match="different_prediction_identity"):
            commit(tmp_path, plan, changed_result, refs, changed_cost)
    assert before == {p.name: p.read_bytes() for p in (tmp_path / "predictions/development_case").iterdir() if p.is_file()}


def test_pending_hard_interruption_retries_identically_and_blocks_other_identity(tmp_path, monkeypatch):
    class HardInterruption(BaseException):
        pass
    _, plan, result, refs, cost, _ = fixture(tmp_path)
    original = store._write_identical_or_new
    def interrupt(path, body):
        if path.name == "receipt.json":
            raise HardInterruption()
        return original(path, body)
    with monkeypatch.context() as patcher:
        patcher.setattr(store, "_write_identical_or_new", interrupt)
        with pytest.raises(HardInterruption):
            commit(tmp_path, plan, result, refs, cost)
    output = tmp_path / "predictions/development_case"
    assert not (output / "receipt.json").exists() and (output / ".pending_identity.json").exists()
    with pytest.raises(store.PredictionStoreError, match="pending_destination"):
        commit(tmp_path, plan, result, refs, dict(cost, optimizer_updates=9))
    resumed = commit(tmp_path, plan, result, refs, cost)
    assert store.verify_prediction_receipt(tmp_path, resumed["receipt"])["status"] == "PREDICTIONS_COMMITTED"


@pytest.mark.parametrize("field", ["logits", "probabilities", "labels", "event_uids", "protocol_freeze_sha"])
def test_prediction_values_must_share_logits_argmax_softmax_and_unique_episode_order(tmp_path, field):
    _, plan, result, refs, cost, _ = fixture(tmp_path)
    if field == "logits":
        value = result.logits.clone(); value[0, 0] = float("nan")
    elif field == "probabilities":
        value = result.probabilities.flip(1)
    elif field == "labels":
        value = result.labels.clone(); value[0] = 1  # Wrong exact-tie policy.
    elif field == "event_uids":
        value = (result.event_uids[0],) + result.event_uids[:-1]
    else:
        value = "b" * 64
    with pytest.raises(store.PredictionStoreError):
        commit(tmp_path, plan, replace(result, **{field: value}), refs, cost)
    assert not (tmp_path / "predictions").exists()


@pytest.mark.parametrize("key,value", [("acquired_calibration_trials", 19), ("valid_support_trials", 20),
    ("accessible_labels", 20), ("label_access_event_uids", []), ("optimizer_updates", -1),
    ("calibration_wall_seconds", float("nan")), ("audit_trials", 1)])
def test_invalid_calibration_count_label_access_or_duration_ledgers_refuse_commit(tmp_path, key, value):
    _, plan, result, refs, cost, _ = fixture(tmp_path)
    with pytest.raises((store.PredictionStoreError, ValueError)):
        commit(tmp_path, plan, result, refs, dict(cost, **{key: value}))
    assert not (tmp_path / "predictions").exists()


def test_no_access_to_invalid_prefix_labels_or_future_labels_and_frozen_can_use_no_labels(tmp_path):
    _, plan, result, refs, cost, _ = fixture(tmp_path)
    for uid in (plan.acquired_event_uids[0], plan.query_event_uids[0]):
        changed = dict(cost, label_access_event_uids=[uid], accessible_labels=1)
        with pytest.raises(store.PredictionStoreError):
            commit(tmp_path, plan, result, refs, changed)
    frozen_cost = dict(cost, accessible_labels=0, label_access_event_uids=[], optimizer_updates=0)
    configuration = tmp_path / refs["configuration"]["path"]
    write(configuration, {"method": "B0", "test_double_not_research_evidence": True})
    refs["configuration"] = descriptor(tmp_path, configuration)
    saved = store.commit_predictions(tmp_path, "predictions/frozen", result, plan, method="B0", pipeline_seed=20261018,
                                    references=refs, cost=frozen_cost)
    assert store.verify_prediction_receipt(tmp_path, saved["receipt"])["method"] == "B0"


def test_external_roles_paths_credentials_and_symlinks_are_rejected_before_creation(tmp_path):
    _, plan, result, refs, cost, _ = fixture(tmp_path)
    for role in ("meta_train", "external_query", "HSStroke2026_v6", "Wang2026"):
        with pytest.raises(PermissionError):
            commit(tmp_path, replace(plan, role=role), result, refs, cost)
    for output in ("../outside", "/tmp/outside", "predictions/HSStroke2026_v6", "predictions/Wang2026",
                   "predictions/external_query", "predictions/sealed", "predictions/.env/token"):
        with pytest.raises(store.PredictionStoreError):
            commit(tmp_path, plan, result, refs, cost, output=output)
    broken = copy.deepcopy(refs)
    broken["model"] = {"path": "data/external/clinical_query.npy", "sha256": "a"*64}
    with pytest.raises(store.PredictionStoreError, match="external_artifact"):
        commit(tmp_path, plan, result, broken, cost)
    (tmp_path / "predictions").symlink_to(tmp_path / "runs", target_is_directory=True)
    with pytest.raises(store.PredictionStoreError, match="symlink"):
        commit(tmp_path, plan, result, refs, cost)


def test_corrupted_committed_array_or_changed_reference_is_preserved_and_not_recommitted(tmp_path):
    _, plan, result, refs, cost, _ = fixture(tmp_path)
    saved = commit(tmp_path, plan, result, refs, cost)
    output = tmp_path / "predictions/development_case"
    (output / "logits.npy").write_bytes(b"corrupted bytes must stay for inspection")
    with pytest.raises(store.PredictionStoreError, match="sha256"):
        store.verify_prediction_receipt(tmp_path, saved["receipt"])
    with pytest.raises(store.PredictionStoreError):
        commit(tmp_path, plan, result, refs, cost)
    assert (output / "logits.npy").read_bytes() == b"corrupted bytes must stay for inspection"


def test_internal_explicit_access_and_committed_fields_unlock_existing_evaluation_labels_only_after_commit(tmp_path):
    _, plan, result, refs, cost, access = fixture(tmp_path, internal=True)
    configuration_sha = refs["configuration"]["sha256"]
    labels = EvaluationLabels(plan.query_event_uids, "internal_holdout", plan.temporal_audit_sha256,
        torch.arange(len(plan.query_event_uids)) % 2, tmp_path, configuration_sha)
    with pytest.raises(PermissionError):
        labels.labels(purpose="internal_development_check")
    for kwargs in ({}, {"purpose": "internal_development_check"},
                   {"purpose": "internal_development_check", "internal_access_receipt": access,
                    "selected_configuration_sha256": "b"*64}):
        with pytest.raises(PermissionError):
            commit(tmp_path, plan, result, refs, cost, **kwargs)
    with patch.object(EvaluationLabels, "labels", side_effect=AssertionError("no label access while committing")):
        saved = commit(tmp_path, plan, result, refs, cost, purpose="internal_development_check",
            internal_access_receipt=access, selected_configuration_sha256=configuration_sha)
    actual = labels.labels(purpose="internal_development_check", prediction_receipt=saved["receipt"])
    assert torch.equal(actual, torch.arange(len(plan.query_event_uids)) % 2)
    with pytest.raises(PermissionError):
        replace(labels, _internal_configuration_sha256="b"*64).labels(
            purpose="internal_development_check", prediction_receipt=saved["receipt"])


def test_store_verification_reads_only_metadata_and_its_own_prediction_arrays(tmp_path, monkeypatch):
    _, plan, result, refs, cost, _ = fixture(tmp_path)
    original = np.load
    seen = []
    def only_prediction_arrays(path, **kwargs):
        path = Path(path)
        assert path.name in {"logits.npy", "probabilities.npy", "predictions.npy"}
        assert kwargs.get("allow_pickle") is False
        seen.append(path.name)
        return original(path, **kwargs)
    monkeypatch.setattr(np, "load", only_prediction_arrays)
    saved = commit(tmp_path, plan, result, refs, cost)
    store.verify_prediction_receipt(tmp_path, saved["receipt"])
    assert set(seen) == {"logits.npy", "probabilities.npy", "predictions.npy"}


def test_rehashed_plan_cannot_move_boundary_run_trial_into_future_query(tmp_path):
    _, plan, result, refs, cost, _ = fixture(tmp_path)
    query = (plan.unused_boundary_run_uids[0],) + plan.query_event_uids[1:]
    audit = json.loads(plan.temporal_audit_json)
    audit["query_uids"] = list(query)
    # Neither a re-signed temporal audit nor a matching prediction UID list can
    # replace the independent original-run/prefix chronology checks.
    plan = replace(plan, query_event_uids=query, temporal_audit_sha256=digest(audit),
                   temporal_audit_json=json.dumps(audit, sort_keys=True, separators=(",", ":")))
    with pytest.raises(store.PredictionStoreError):
        commit(tmp_path, plan, replace(result, event_uids=query), refs, cost)
    assert not (tmp_path / "predictions").exists()


def test_pending_payload_symlink_is_preserved_and_never_adopted(tmp_path, monkeypatch):
    class HardInterruption(BaseException):
        pass
    _, plan, result, refs, cost, _ = fixture(tmp_path)
    original = store._write_identical_or_new
    def interrupt(path, body):
        if path.name == "receipt.json":
            raise HardInterruption()
        return original(path, body)
    with monkeypatch.context() as patcher:
        patcher.setattr(store, "_write_identical_or_new", interrupt)
        with pytest.raises(HardInterruption):
            commit(tmp_path, plan, result, refs, cost)
    output = tmp_path / "predictions/development_case"
    target = tmp_path / refs["model"]["path"]
    before = target.read_bytes()
    (output / "logits.npy").unlink()
    (output / "logits.npy").symlink_to(target)
    with pytest.raises(store.PredictionStoreError, match="symlink"):
        commit(tmp_path, plan, result, refs, cost)
    assert (output / "logits.npy").is_symlink() and target.read_bytes() == before
    assert not (output / "receipt.json").exists()


def test_absolute_path_output_inside_root_is_supported_without_allowing_escape(tmp_path):
    _, plan, result, refs, cost, _ = fixture(tmp_path)
    saved = commit(tmp_path, plan, result, refs, cost, output=tmp_path / "runs/predictions/synthetic_case")
    assert saved["receipt"]["path"] == "runs/predictions/synthetic_case/receipt.json"
    assert store.verify_prediction_receipt(tmp_path, saved["receipt"])["status"] == "PREDICTIONS_COMMITTED"


def test_source_pipeline_seed_and_declared_method_cannot_be_relabelled(tmp_path):
    _, plan, result, refs, cost, _ = fixture(tmp_path)
    with pytest.raises(store.PredictionStoreError, match="pipeline_seed_mismatch"):
        store.commit_predictions(tmp_path, "predictions/different_seed", result, plan,
            method="P", pipeline_seed=20261019, references=refs, cost=cost)
    with pytest.raises(store.PredictionStoreError, match="configuration_method"):
        store.commit_predictions(tmp_path, "predictions/different_method", result, plan,
            method="B1", pipeline_seed=20261018, references=refs, cost=cost)
    assert not (tmp_path / "predictions").exists()


def test_wrong_events_or_raw_reference_is_rejected_before_its_body_is_read(tmp_path, monkeypatch):
    native, plan, result, refs, cost, _ = fixture(tmp_path)
    events = native.session(plan.participant_id)["events"]
    raw = tmp_path / "runs/prediction_fixture/not_a_model.edf"
    raw.write_bytes(b"synthetic raw-like fixture, never a model or real EEG")
    raw_ref = descriptor(tmp_path, raw)
    forbidden = {tmp_path / events["path"], raw}
    original_digest = store.file_digest
    def safe_digest(path):
        assert Path(path) not in forbidden, "incorrect refs must fail before reading any event/raw body"
        return original_digest(path)
    monkeypatch.setattr(store, "file_digest", safe_digest)
    broken = dict(refs, episode_manifest=events)
    with pytest.raises(store.PredictionStoreError, match="receipt_bound_episode"):
        commit(tmp_path, plan, result, broken, cost)
    with pytest.raises(store.PredictionStoreError, match="type_mismatch"):
        commit(tmp_path, plan, result, dict(refs, model=raw_ref), cost)
    assert not (tmp_path / "predictions").exists()
