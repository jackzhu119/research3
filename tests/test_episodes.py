"""Synthetic software fixtures only: no real EEG, training, R2 or GPU execution.

Official-shaped metadata and status records below are test doubles for gate
math. They are not empirical evidence that W1/W3 or any research run completed.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from p3work.adapt import CalibrationPrefix
from p3work.contracts import CHANNELS, digest, file_digest, validate_signal_contract
from p3work.episodes import (
    DeterministicEpisodeSampler, EpisodeGateError, EvaluationEpisode, EvaluationLabels,
    load_episode_catalog,
)
from p3work.methods.meta_train import MetaTrainEpisode, episode_objective
from p3work.splits import freeze_subject_split


REPO = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def descriptor(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}


def array_sha(value):
    result = hashlib.sha256(str(value.shape).encode() + str(value.dtype).encode())
    result.update(np.ascontiguousarray(value).tobytes())
    return result.hexdigest()


class EpisodeFixture:
    def __init__(self, root):
        self.root = root
        roster = [{"participant_id": f"S{i}", "identity_verified": True,
                   "clinic11_present_verified": True, "pure_lr_present_verified": True,
                   "intervention_group": None} for i in range(1, 63)]
        self.split = freeze_subject_split(roster)
        self.train = tuple(self.split["partitions"]["meta_train"])
        self.validation = tuple(self.split["partitions"]["meta_validation"])
        self.internal = tuple(self.split["partitions"]["internal_holdout"])
        self.split_path = root / "manifests/participant_split.json"
        write(self.split_path, self.split)
        self.contract = {
            "status": "window_frozen", "channels": list(CHANNELS), "sfreq_hz": 128,
            "selected_profile": "long", "context_s": [0, 3], "crop_s": [.5, 2.5], "n_times": 256,
            "protocol_freeze_sha256": None,
            "window_audit_receipt": {"decision_pool_role": "meta_train",
                "split_sha256": self.split["split_sha256"], "raw_metadata_manifest_sha256": "a" * 64,
                "long": {"participant_mean_coverage": .95, "participant_mean_abs_class_exclusion_gap": .01},
                "short": {"participant_mean_coverage": .99, "participant_mean_abs_class_exclusion_gap": .01}},
        }
        self.contract_path = root / "contracts/input_contract.json"
        write(self.contract_path, self.contract)
        self.contract_sha = validate_signal_contract(self.contract)
        for code in ("p3work/adapters/stieger_native.py", "p3work/preprocess.py"):
            target = root / code
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((REPO / code).read_bytes())
        self.adapter = descriptor(root, root / "p3work/adapters/stieger_native.py")
        self.processing = descriptor(root, root / "p3work/preprocess.py")
        raw = b"synthetic software fixture: not a real MAT recording"
        self.raw_sha = hashlib.sha256(raw).hexdigest()
        md5 = hashlib.md5(raw).hexdigest()
        # 62 base sessions, one extra training session, and 535 unaccessed
        # internal metadata-only registrations: 598 total fixture objects.
        identities = [(f"S{i}", 1) for i in range(1, 63)]
        identities += [(self.train[0], 2)]
        identities += [(self.internal[0], session) for session in range(2, 537)]
        assert len(identities) == 598
        files = [{"id": index + 1, "name": f"{person}_Session_{session}.mat", "size": len(raw),
                  "provider_subject": person, "session_ordinal": session,
                  "download_url": f"https://ndownloader.figshare.com/files/{index+1}",
                  "computed_md5": md5, "supplied_md5": md5}
                 for index, (person, session) in enumerate(identities)]
        inventory_path = root / "data_catalog/stieger2021/official_inventory.json"
        write(inventory_path, {"article_id": 13123148, "version": 1,
                              "license": {"name": "CC BY 4.0"}, "files": files})
        values = np.random.default_rng(77).normal(size=(75, 11, 256)) * 1e-6
        values -= values.mean(axis=1, keepdims=True)
        values -= values.mean(axis=2, keepdims=True)
        self.values = values.astype(np.float32)
        shared_x = root / "cache/stieger/shared_test_double_X.npy"
        shared_x.parent.mkdir(parents=True, exist_ok=True)
        np.save(shared_x, self.values[1:], allow_pickle=False)
        partition = {person: role for role, people in self.split["partitions"].items() for person in people}
        self.sessions = []
        for entry in files:
            person, session = entry["provider_subject"], str(entry["session_ordinal"])
            role = partition[person]
            target = root / "cache/stieger" / role / person / session / "events.json"
            record = {"participant_id": person, "session_id": session, "partition": role,
                      "source_record_id": entry["name"], "source_file_sha256": self.raw_sha,
                      "source_file_bytes": len(raw), "native_sfreq_hz": 1000,
                      "raw_integrity": {"sha256": self.raw_sha, "bytes": len(raw), "md5": md5},
                      "acquisition_order_verified": True, "original_lr_trials": 75, "valid_trials": 74,
                      "X": descriptor(root, shared_x)}
            if role == "internal_holdout":
                # Deliberately absent: a development loader must never try to
                # open this payload when requesting train/validation partitions.
                record["events"] = {"path": target.relative_to(root).as_posix(), "sha256": "b" * 64}
            else:
                write(target, self.rows(record))
                record["events"] = descriptor(root, target)
            self.sessions.append(record)
        self.manifest_path = root / "cache/stieger/stieger_cache.json"
        self.manifest = {"schema_version": "paper3-stieger-cache-v2", "status": "STIEGER_CACHE_COMPLETE",
            "dataset": "Stieger2021", "synthetic": False, "old_npz_reused": False,
            "test_double_not_research_evidence": True,
            "stage": "pre_rms_car11_temporal_demeaned", "normalization_applied": False,
            "partition_scope": ["meta_train", "meta_validation", "internal_holdout"],
            "input_contract_sha256": self.contract_sha, "input_contract": descriptor(root, self.contract_path),
            "participant_split": descriptor(root, self.split_path), "split_sha256": self.split["split_sha256"],
            "adapter_code": self.adapter, "processing_code": self.processing,
            "official_inventory": descriptor(root, inventory_path), "sessions": self.sessions}
        self.tasks = {"W1": {"status": "COMPLETE"}, "W3": {"status": "COMPLETE"},
                      "W7": {"status": "NOT_FROZEN", "protocol_freeze_sha256": None}}
        self.refresh()

    def rows(self, session):
        rows = []
        for index in range(75):
            valid = index != 0
            uid = (f"Stieger2021:v1:{session['participant_id']}:session{session['session_id']}:"
                   f"{self.raw_sha}:trial{index}")
            label = index % 2
            rows.append({"event_uid": uid, "trial_uid": uid, "dataset": "Stieger2021", "dataset_version": 1,
                "participant_id": session["participant_id"], "session_id": session["session_id"],
                "partition": session["partition"],
                "source_record_id": session["source_record_id"], "source_file_sha256": self.raw_sha,
                "original_trial_index": index, "provider_trialnumber": index % 25 + 1,
                "physical_run_id": index // 25 + 1, "physical_run_complete_verified": True,
                "tasknumber": 1, "label": label, "canonical_label": "right_hand" if label else "left_hand",
                "native_sfreq_hz": 1000, "acquisition_order_verified": True,
                "input_contract_sha256": self.contract_sha, "adapter_sha256": self.adapter["sha256"],
                "processing_sha256": self.processing["sha256"], "valid": valid,
                "cache_index": index - 1 if valid else None,
                "exclusion_reason": None if valid else "insufficient_original_feedback_duration",
                "analysis_anchor_sample": 2000, "context_start_sample": 2000, "context_stop_sample": 5000,
                "feedback_stop_exclusive_sample": 5500 if valid else 3000,
                "native_trial_samples_including_padding": 6500 if valid else 4000,
                "feedback_end_verified": True,
                "signal_sha256": array_sha(self.values[index]) if valid else None,
                "raw_context_sha256": "a" * 64 if valid else None,
                "preprocessing_qc": {"signal_contract_sha256": self.contract_sha, "normalization_applied": False,
                    "data_after_context_accessed": False, "filter_context_ends_at_native_sample": 5000}})
        return rows

    def refresh(self):
        write(self.manifest_path, self.manifest)
        self.tasks["W3"]["cache_manifest"] = descriptor(self.root, self.manifest_path)
        write(self.root / "status/tasks.json", self.tasks)

    def session(self, person=None, session="1"):
        person = person or self.train[0]
        return next(row for row in self.sessions if row["participant_id"] == person and row["session_id"] == session)

    def mutate_rows(self, person, change, session="1"):
        item = self.session(person, session)
        path = self.root / item["events"]["path"]
        rows = json.loads(path.read_text())
        change(rows)
        write(path, rows)
        item["events"] = descriptor(self.root, path)
        self.refresh()

    def load(self, role="meta_train", **kwargs):
        return load_episode_catalog(self.root, role=role, **kwargs)


@pytest.fixture
def fixture(tmp_path):
    return EpisodeFixture(tmp_path)


def test_B20_cost_retains_invalid_and_excludes_entire_boundary_run(fixture):
    catalog = fixture.load()
    plan = catalog.plan(fixture.train[0], "1")
    assert plan.acquired_calibration_trials == 20
    assert len(plan.support_event_uids) == 19
    assert len(plan.unused_boundary_run_uids) == 5
    assert len(plan.query_event_uids) == 50
    assert set(plan.query_event_uids).isdisjoint(plan.acquired_event_uids + plan.unused_boundary_run_uids)
    assert plan.acquired_event_uids[0] not in plan.support_event_uids
    episode = catalog.materialize(plan)
    assert type(episode) is MetaTrainEpisode
    assert type(episode.support) is CalibrationPrefix
    assert len(episode.support.acquired_event_uids) == 20
    assert episode.support.x.shape == (19, 11, 256)
    assert episode.query_x.dtype == torch.float64
    assert episode.query_y.dtype == torch.long
    assert episode.support.protocol_freeze_sha == fixture.contract_sha
    assert catalog.final_protocol_freeze_sha256 is None
    audit = plan.to_dict()["temporal_audit"]
    assert digest(audit) == plan.temporal_audit_sha256
    assert len(audit["original_event_audit"]) == 75
    assert audit["original_event_audit"][0]["valid"] is False
    assert audit["original_event_audit"][20]["physical_run_id"] == 1
    assert audit["original_event_audit"][25]["physical_run_id"] == 2
    assert all("label" not in row for row in audit["original_event_audit"])


def test_complete_future_runs_are_merged_before_per_class_query_threshold(fixture):
    def one_class_per_run(rows):
        for row in rows[25:]:
            row["label"] = 0 if row["physical_run_id"] == 2 else 1
            row["canonical_label"] = "right_hand" if row["label"] else "left_hand"
    fixture.mutate_rows(fixture.train[0], one_class_per_run)
    catalog = fixture.load()
    episode = catalog.materialize(catalog.plan(fixture.train[0], "1"))
    assert episode.query_y.bincount(minlength=2).tolist() == [25, 25]


def test_incomplete_future_run_is_removed_as_whole_run(fixture):
    def incomplete(rows):
        for row in rows[25:50]:
            row["physical_run_complete_verified"] = False
    fixture.mutate_rows(fixture.train[0], incomplete)
    catalog = fixture.load()
    plan = catalog.plan(fixture.train[0], "1")
    assert len(plan.incomplete_future_run_uids) == 25
    assert len(plan.query_event_uids) == 25
    assert plan.query_eligible


def test_missing_support_class_is_preserved_and_sampled_for_f0_fallback(fixture):
    session = fixture.session()
    path = fixture.root / session["events"]["path"]
    rows = json.loads(path.read_text())
    for row in rows[:20]:
        row["valid"] = row["label"] == 0
        row["exclusion_reason"] = None if row["valid"] else "synthetic_fixed_QC_exclusion"
    valid = [row for row in rows if row["valid"]]
    for index, row in enumerate(valid):
        row["cache_index"] = index
    for row in rows:
        if not row["valid"]:
            row["cache_index"] = None
    # The formerly short original stays short and invalid; no trial is padded.
    rows[0]["valid"] = False
    rows[0]["cache_index"] = None
    rows[0]["exclusion_reason"] = "insufficient_original_feedback_duration"
    valid = [row for row in rows if row["valid"]]
    for index, row in enumerate(valid):
        row["cache_index"] = index
    x_path = path.with_name("fallback_X.npy")
    np.save(x_path, fixture.values[[row["original_trial_index"] for row in valid]], allow_pickle=False)
    session["X"] = descriptor(fixture.root, x_path)
    session["valid_trials"] = len(valid)
    write(path, rows)
    session["events"] = descriptor(fixture.root, path)
    fixture.refresh()
    catalog = fixture.load()
    plan = catalog.plan(fixture.train[0], "1")
    assert plan.fallback_reason == "insufficient_class_support"
    assert plan.query_eligible
    sampler = DeterministicEpisodeSampler(catalog, 7, participant_subset=[fixture.train[0]])
    assert plan in sampler._eligible[fixture.train[0]]
    episode = catalog.materialize(plan)
    assert set(episode.support.y.tolist()) == {0}
    assert len(episode.support.acquired_event_uids) == 20
    assert len(episode.support.event_uids) == 9


def test_query_threshold_failure_is_recorded_and_no_person_is_silently_reassigned(fixture):
    person = fixture.train[1]  # Only one registered session for this person.
    def no_right_query(rows):
        for row in rows[25:]:
            row["label"] = 0
            row["canonical_label"] = "left_hand"
    fixture.mutate_rows(person, no_right_query)
    catalog = fixture.load()
    assert not catalog.plan(person, "1").query_eligible
    with pytest.raises(EpisodeGateError, match="not_evaluable"):
        catalog.materialize(catalog.plan(person, "1"))
    with pytest.raises(EpisodeGateError, match="no_silent_reassignment"):
        DeterministicEpisodeSampler(catalog, 4)


def test_sampler_is_subject_uniform_then_session_uniform_and_exactly_resumable(fixture):
    catalog = fixture.load()
    sampler = DeterministicEpisodeSampler(catalog, 20261018)
    counts = {person: 0 for person in fixture.train}
    first_sessions = {"1": 0, "2": 0}
    for _ in range(8000):
        plan = sampler.next_plan()
        counts[plan.participant_id] += 1
        if plan.participant_id == fixture.train[0]:
            first_sessions[plan.session_id] += 1
    # A trial- or session-weighted sampler would double this two-session
    # participant's frequency. Fixed draws keep every person near 8000/40.
    assert all(120 <= count <= 280 for count in counts.values())
    assert all(count >= 40 for count in first_sessions.values())
    checkpoint = sampler.state_dict()
    expected = [sampler.next_plan() for _ in range(20)]
    restored = DeterministicEpisodeSampler(catalog, 20261018)
    restored.load_state_dict(json.loads(json.dumps(checkpoint)))
    assert [restored.next_plan() for _ in range(20)] == expected
    bad = dict(checkpoint, role="meta_validation")
    with pytest.raises(EpisodeGateError, match="resume_data_split_role_or_seed"):
        restored.load_state_dict(bad)
    wrong_seed = DeterministicEpisodeSampler(catalog, 1)
    with pytest.raises(EpisodeGateError, match="resume_data_split_role_or_seed"):
        wrong_seed.load_state_dict(checkpoint)
    assert type(restored.next()) is MetaTrainEpisode


def test_validation_has_explicit_evaluation_labels_and_cannot_enter_meta_gradient(fixture):
    catalog = fixture.load(role="meta_validation")
    assert all(plan.participant_id in fixture.validation for plan in catalog.plans)
    episode = catalog.materialize(catalog.plans[0])
    assert type(episode) is EvaluationEpisode
    assert episode.support.role == "meta_validation_support"
    assert not hasattr(episode, "query_y")
    assert episode.evaluation_labels.labels().shape == (50,)
    assert torch.equal(episode.evaluation_labels(), episode.evaluation_labels.labels())
    with pytest.raises(PermissionError, match="cannot_supply_meta_gradients"):
        episode.evaluation_labels.labels(purpose="meta_gradient")
    with pytest.raises(PermissionError, match="meta_train_catalog"):
        DeterministicEpisodeSampler(catalog, 1)
    with pytest.raises(TypeError, match="MetaTrainEpisode capability"):
        episode_objective(None, None, None, episode)
    with pytest.raises(PermissionError, match="session_not_in_authorized_partition"):
        catalog.plan(fixture.train[0], "1")


def test_internal_and_external_roles_fail_before_signal_payload_access(fixture, monkeypatch):
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("unauthorized signal read"))
    with pytest.raises(PermissionError, match="explicit_post_selection_access_receipt"):
        fixture.load(role="internal_holdout")
    with pytest.raises(PermissionError, match="development_partitions"):
        fixture.load(role="HSStroke2026_v6")
    with pytest.raises(PermissionError, match="partition_purpose"):
        fixture.load(role="meta_validation", purpose="meta_gradient")


@pytest.mark.parametrize("stage", ["W1", "W3"])
def test_unfinished_stage_fails_before_numerical_cache_access(fixture, monkeypatch, stage):
    fixture.tasks[stage]["status"] = "PARTIAL"
    fixture.refresh()
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("premature numerical read"))
    with pytest.raises(EpisodeGateError, match="must_be_COMPLETE"):
        fixture.load()


@pytest.mark.parametrize("field,value", [("synthetic", True), ("old_npz_reused", True),
                                        ("status", "PARTIAL_CACHE"), ("normalization_applied", True)])
def test_non_real_or_unfrozen_cache_is_rejected(fixture, field, value):
    fixture.manifest[field] = value
    fixture.refresh()
    with pytest.raises(EpisodeGateError, match="complete_real_frozen"):
        fixture.load()


def test_actual_manifest_code_split_and_selected_array_hashes_are_enforced(fixture):
    catalog = fixture.load()
    plan = catalog.plans[0]
    x_path = fixture.root / fixture.session()["X"]["path"]
    with x_path.open("ab") as stream:
        stream.write(b"changed after verification")
    with pytest.raises(EpisodeGateError, match="artifact_sha256"):
        catalog.materialize(plan)
    code = fixture.root / "p3work/preprocess.py"
    code.write_text(code.read_text() + "\n# changed\n")
    with pytest.raises(EpisodeGateError, match="artifact_sha256"):
        fixture.load()


@pytest.mark.parametrize("change", [
    lambda rows: rows[1].update(context_stop_sample=6000),
    lambda rows: rows[1].update(feedback_stop_exclusive_sample=4500),
    lambda rows: rows[1].update(tasknumber=3),
    lambda rows: rows[1].update(source_file_sha256="0" * 64),
    lambda rows: rows[1].update(cache_index=70),
    lambda rows: rows[1].update(acquisition_order_verified=False),
    lambda rows: rows[1].update(provider_trialnumber=2_000_000),
])
def test_true_original_source_bounds_task_order_and_run_evidence_are_required(fixture, change):
    fixture.mutate_rows(fixture.train[0], change)
    with pytest.raises(EpisodeGateError):
        fixture.load()


def test_split_partition_tampering_cannot_transfer_validation_person_to_training(fixture):
    fixture.session(fixture.validation[0])["partition"] = "meta_train"
    fixture.refresh()
    with pytest.raises(EpisodeGateError, match="official_split_registered"):
        fixture.load()


def test_different_manifest_cannot_bypass_W3_authority(fixture):
    fake = {"path": "unbound.json", "sha256": "0" * 64}
    with pytest.raises(EpisodeGateError, match="not_bound_to_W3"):
        load_episode_catalog(fixture.root, cache_manifest=fake)


def test_foreign_role_plan_cannot_be_materialized_as_training(fixture):
    train = fixture.load()
    validation = fixture.load(role="meta_validation")
    with pytest.raises(PermissionError, match="not_issued_by_this_catalog"):
        train.materialize(validation.plans[0])


def test_unverified_continuous_fallback_cannot_replace_unknown_physical_runs(fixture):
    def erase_runs(rows):
        for row in rows:
            row["physical_run_id"] = None
    fixture.mutate_rows(fixture.train[0], erase_runs)
    with pytest.raises(EpisodeGateError, match="verified_continuous_original_trial_block"):
        fixture.load()
    fixture.session()["continuous_original_trial_block_verified"] = True
    fixture.refresh()
    catalog = fixture.load()
    plan = catalog.plan(fixture.train[0], "1")
    assert plan.query_rule == "verified_continuous_original_trial_block"
    assert len(plan.query_event_uids) == 55
    assert plan.unused_boundary_run_uids == ()


def test_unverified_invalid_original_keeps_B20_acquisition_cost(fixture):
    def unverified(rows):
        rows[0].update(feedback_end_verified=False, feedback_stop_exclusive_sample=999999,
                       exclusion_reason="insufficient_or_unverified_real_feedback")
    fixture.mutate_rows(fixture.train[0], unverified)
    catalog = fixture.load()
    plan = catalog.plan(fixture.train[0], "1")
    assert len(plan.acquired_event_uids) == 20
    assert len(plan.support_event_uids) == 19
    assert plan.acquired_event_uids[0] not in plan.support_event_uids


def test_actual_signal_hash_binds_cache_index_to_original_event(fixture):
    fixture.mutate_rows(fixture.train[0], lambda rows: rows[1].update(signal_sha256="f" * 64))
    with pytest.raises(EpisodeGateError, match="event_to_actual_signal"):
        fixture.load()


def test_frozen_split_content_hash_is_checked_after_file_hash(fixture):
    changed = copy.deepcopy(fixture.split)
    changed["partitions"]["meta_train"][0], changed["partitions"]["meta_validation"][0] = (
        changed["partitions"]["meta_validation"][0], changed["partitions"]["meta_train"][0])
    write(fixture.split_path, changed)
    fixture.manifest["participant_split"] = descriptor(fixture.root, fixture.split_path)
    fixture.refresh()
    with pytest.raises(EpisodeGateError, match="frozen_participant_split"):
        fixture.load()


def synthetic_internal_prediction_transaction(tmp_path):
    # Reuse the prediction store's explicit synthetic transaction fixture. It
    # maps a test plan to an internal metadata double without reading internal
    # signal/event payloads or performing inference, adaptation or fitting.
    from tests.test_prediction_store import fixture as prediction_fixture, commit
    _, plan, result, references, cost, access = prediction_fixture(tmp_path, internal=True)
    configuration_sha = references["configuration"]["sha256"]
    labels = EvaluationLabels(
        plan.query_event_uids, "internal_holdout", plan.temporal_audit_sha256,
        torch.arange(len(plan.query_event_uids), dtype=torch.long) % 2,
        tmp_path, configuration_sha,
    )
    saved = commit(tmp_path, plan, result, references, cost, purpose="internal_development_check",
                   internal_access_receipt=access, selected_configuration_sha256=configuration_sha)
    return plan, labels, saved


def test_internal_labels_reject_self_hashed_five_field_receipt_without_transaction(tmp_path):
    from tests.test_prediction_store import fixture as prediction_fixture
    _, plan, _, references, _, _ = prediction_fixture(tmp_path, internal=True)
    configuration_sha = references["configuration"]["sha256"]
    labels = EvaluationLabels(
        plan.query_event_uids, "internal_holdout", plan.temporal_audit_sha256,
        torch.arange(len(plan.query_event_uids), dtype=torch.long) % 2, tmp_path, configuration_sha,
    )
    forged = tmp_path / "predictions/forged_self_hashed_receipt.json"
    write(forged, {"status": "PREDICTIONS_COMMITTED", "episode_plan_sha256": plan.temporal_audit_sha256,
        "query_event_uids": list(plan.query_event_uids), "labels_accessed_before_predictions": False,
        "selected_configuration_sha256": configuration_sha})
    with pytest.raises(PermissionError, match="transaction_verification_failed"):
        labels.labels(purpose="internal_development_check", prediction_receipt=descriptor(tmp_path, forged))


def test_internal_labels_unlock_only_after_whole_synthetic_prediction_transaction_verifies(tmp_path):
    plan, labels, saved = synthetic_internal_prediction_transaction(tmp_path)
    values = labels.labels(purpose="internal_development_check", prediction_receipt=saved["receipt"])
    assert values.tolist() == (torch.arange(len(plan.query_event_uids)) % 2).tolist()
    values.zero_()
    # The caller cannot mutate the evaluator's held label view through a result.
    assert int(labels.labels(purpose="internal_development_check", prediction_receipt=saved["receipt"]).sum()) > 0


@pytest.mark.parametrize("name", ["logits.npy", "probabilities.npy", "predictions.npy"])
@pytest.mark.parametrize("change", ["tamper", "delete"])
def test_internal_labels_remain_locked_after_committed_prediction_payload_changes(tmp_path, name, change):
    _, labels, saved = synthetic_internal_prediction_transaction(tmp_path)
    path = (tmp_path / saved["receipt"]["path"]).with_name(name)
    if change == "delete":
        path.unlink()
    else:
        path.write_bytes(b"corrupted synthetic prediction payload must not unlock labels")
    with pytest.raises(PermissionError, match="transaction_verification_failed"):
        labels.labels(purpose="internal_development_check", prediction_receipt=saved["receipt"])
