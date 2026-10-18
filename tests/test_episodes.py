"""Synthetic software fixtures only: no real EEG, training, R2 or GPU execution.

Official-shaped metadata and status records below are test doubles for gate
math. They are not empirical evidence that W1/W3 or any research run completed.
"""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from p3work.adapt import CalibrationPrefix
from p3work import episodes as episode_module
from p3work.contracts import CHANNELS, digest, file_digest, validate_signal_contract
from p3work.episodes import (
    DeterministicEpisodeSampler, EpisodeGateError, EvaluationEpisode, EvaluationLabels,
    VerifiedEpisodeCatalog, load_episode_catalog,
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
                "original_trial_index": index, "provider_trialnumber": index + 1,
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
        self.tasks["W1"]["input_contract_sha256"] = self.contract_sha
        self.tasks["W3"].update(input_contract_sha256=self.contract_sha, split_sha256=self.split["split_sha256"])
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


@pytest.mark.parametrize("role", ["meta_train", "meta_validation"])
def test_support_only_matches_full_episode_bitwise_and_keeps_original_B20_cost(fixture, role):
    catalog = fixture.load(role=role); plan = catalog.plans[0]
    expected = catalog.materialize(plan).support
    actual = catalog.materialize_support(plan, device="cpu")
    assert type(actual) is CalibrationPrefix
    assert torch.equal(actual.x, expected.x) and torch.equal(actual.y, expected.y)
    assert actual.x.dtype == torch.float64 and actual.y.dtype == torch.long
    assert actual.x.device.type == actual.y.device.type == "cpu"
    assert actual.event_uids == expected.event_uids == plan.support_event_uids
    assert actual.acquired_event_uids == expected.acquired_event_uids == plan.acquired_event_uids
    assert len(actual.acquired_event_uids) == 20 and len(actual.event_uids) == 19
    assert actual.acquired_event_uids[0] not in actual.event_uids
    assert actual.role == expected.role and actual.protocol_freeze_sha == expected.protocol_freeze_sha


@pytest.mark.parametrize("role", ["meta_train", "meta_validation"])
def test_support_only_never_indexes_query_signal_or_rows_or_constructs_query_capabilities(fixture, monkeypatch, role):
    catalog = fixture.load(role=role); plan = catalog.plans[0]
    session = catalog._sessions[plan.participant_id, plan.session_id]
    allowed_uids = set(plan.support_event_uids)
    allowed_indices = {session["rows"][uid]["cache_index"] for uid in allowed_uids}

    class PrefixOnlyRows:
        def __init__(self, rows):
            self.rows = rows

        def __getitem__(self, uid):
            assert uid in allowed_uids, "Future or excluded metadata/labels accessed"
            return self.rows[uid]

    real_load = np.load
    real_tensors = episode_module._event_tensors

    def prefix_only_tensors(x, rows, uids, **kwargs):
        assert set(uids) <= allowed_uids, "Future or excluded label view requested"
        return real_tensors(x, PrefixOnlyRows(rows), uids, **kwargs)

    class PrefixOnlyArray:
        def __init__(self, array):
            self.array = array

        def __getitem__(self, indices):
            assert set(indices) <= allowed_indices, "Future or excluded numerical cache indexed"
            return self.array[indices]

    def load_prefix_only(*args, **kwargs):
        assert kwargs == {"mmap_mode": "r", "allow_pickle": False}
        return PrefixOnlyArray(real_load(*args, **kwargs))

    def no_query_capability(*args, **kwargs):
        pytest.fail("Support-only path constructed a query/label capability")

    monkeypatch.setattr(np, "load", load_prefix_only)
    monkeypatch.setattr(episode_module, "_event_tensors", prefix_only_tensors)
    for name in ("PredictionInputs", "EvaluationLabels", "MetaTrainEpisode"):
        monkeypatch.setattr(episode_module, name, no_query_capability)
    support = catalog.materialize_support(plan)
    assert support.event_uids == plan.support_event_uids and len(support.acquired_event_uids) == 20


@pytest.mark.parametrize("payload", ["X", "events"])
def test_support_only_rechecks_actual_array_and_event_hashes(fixture, payload):
    catalog = fixture.load(); plan = catalog.plans[0]
    path = fixture.root / fixture.session()[payload]["path"]
    with path.open("ab") as stream:
        stream.write(b"SYNTHETIC_MUTATION_AFTER_CATALOG_VERIFICATION")
    with pytest.raises(EpisodeGateError, match="artifact_sha256"):
        catalog.materialize_support(plan)


def test_support_only_rejects_unknown_tampered_and_foreign_role_plans_before_payload_access(fixture, monkeypatch):
    catalog = fixture.load(); plan = catalog.plans[0]
    validation = fixture.load(role="meta_validation")
    bad = [object(), validation.plans[0], replace(plan, session_id="999999"),
           replace(plan, role="internal_holdout"), replace(plan, manifest_sha256="0" * 64),
           replace(plan, support_event_uids=plan.query_event_uids[:19]),
           replace(plan, acquired_event_uids=plan.query_event_uids[:20]),
           replace(plan, temporal_audit_sha256="0" * 64)]
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Unknown/foreign plan touched numerical cache"))
    for candidate in bad:
        with pytest.raises(PermissionError, match="not_issued_by_this_catalog"):
            catalog.materialize_support(candidate)


def test_support_only_cannot_repurpose_a_catalog_permission(fixture, monkeypatch):
    catalog = fixture.load(); plan = catalog.plans[0]
    with pytest.raises(AttributeError, match="bindings_are_immutable"):
        catalog.purpose = "internal_development_check"
    # Even a deliberate object.__setattr__ bypass cannot rewrite the external
    # issuance record and acquire a different capability.
    object.__setattr__(catalog, "purpose", "internal_development_check")
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Wrong-purpose catalog touched numerical cache"))
    with pytest.raises(PermissionError, match="binding_changed"):
        catalog.materialize_support(plan)


def test_catalog_nested_bindings_cannot_remap_future_query_rows_into_support(fixture):
    catalog = fixture.load(); plan = catalog.plans[0]
    key = plan.participant_id, plan.session_id
    session = catalog._sessions[key]
    support_uid, query_uid = plan.support_event_uids[0], plan.query_event_uids[0]
    expected = catalog.materialize_support(plan)
    with pytest.raises(TypeError):
        session["rows"] = {support_uid: session["rows"][query_uid]}
    with pytest.raises(TypeError):
        session["rows"][support_uid] = session["rows"][query_uid]
    with pytest.raises(TypeError):
        session["rows"][support_uid]["cache_index"] = session["rows"][query_uid]["cache_index"]
    with pytest.raises(TypeError):
        session["rows"][support_uid]["preprocessing_qc"]["data_after_context_accessed"] = True
    with pytest.raises(TypeError):
        session["X"]["path"] = "future_query.npy"
    with pytest.raises(TypeError):
        catalog._plans[key] = replace(plan, support_event_uids=plan.query_event_uids[:19])
    with pytest.raises(AttributeError):
        plan.support_event_uids = plan.query_event_uids[:19]
    assert not hasattr(plan, "__dict__")
    with pytest.raises(AttributeError, match="bindings_are_immutable"):
        catalog._sessions = {key: session}
    actual = catalog.materialize_support(plan)
    assert torch.equal(actual.x, expected.x) and torch.equal(actual.y, expected.y)


def test_public_constructor_and_unissued_object_cannot_forge_verified_catalog(fixture, monkeypatch):
    catalog = fixture.load(); plan = catalog.plans[0]
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Unissued catalog touched numerical cache"))
    with pytest.raises(PermissionError, match="must_be_issued_by_verified_loader"):
        VerifiedEpisodeCatalog(root=fixture.root, role="meta_train", purpose="meta_gradient",
            manifest_sha256=catalog.manifest_sha256, split_sha256=catalog.split_sha256,
            input_contract_sha256=catalog.input_contract_sha256, n_times=catalog.n_times,
            sessions=catalog._sessions, plans=catalog.plans, partition_people=catalog.partition_people)
    unissued = object.__new__(VerifiedEpisodeCatalog)
    for operation in (lambda: unissued.materialize_support(plan), lambda: unissued.materialize(plan),
                      lambda: unissued.plan(plan.participant_id, plan.session_id),
                      lambda: DeterministicEpisodeSampler(unissued, 1)):
        with pytest.raises(PermissionError, match="no_verified_issuance_seal"):
            operation()


def test_independent_plan_hash_seal_rejects_low_level_rewrite_of_same_issued_object(fixture, monkeypatch):
    catalog = fixture.load(); plan = catalog.plans[0]
    sampler = DeterministicEpisodeSampler(catalog, 1, participant_subset=[plan.participant_id])
    # Fix this software fixture to the first session of this participant.
    monkeypatch.setattr(DeterministicEpisodeSampler, "_draw", lambda self, size, stage: 0)
    object.__setattr__(plan, "support_event_uids", plan.query_event_uids[:19])
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Rewritten issued plan touched numerical cache"))
    for operation in (lambda: catalog.materialize_support(plan), lambda: catalog.materialize(plan),
                      lambda: catalog.plan(plan.participant_id, plan.session_id), sampler.next_plan):
        with pytest.raises(PermissionError, match="issued_episode_plan_binding_changed"):
            operation()
    assert sampler.cursor == 0


@pytest.mark.parametrize("operation", ["support", "episode", "sample"])
@pytest.mark.parametrize("stage", ["W1", "W3"])
def test_current_stage_downgrade_revokes_existing_materialization_and_sampler(fixture, monkeypatch, stage, operation):
    catalog = fixture.load(); plan = catalog.plans[0]
    sampler = DeterministicEpisodeSampler(catalog, 1)
    fixture.tasks[stage]["status"] = "PARTIAL"
    write(fixture.root / "status/tasks.json", fixture.tasks)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Revoked stage touched numerical cache"))
    calls = {"support": lambda: catalog.materialize_support(plan), "episode": lambda: catalog.materialize(plan),
             "sample": sampler.next_plan}
    with pytest.raises(EpisodeGateError, match="must_be_COMPLETE"):
        calls[operation]()
    assert sampler.cursor == 0
    with pytest.raises(EpisodeGateError, match="must_be_COMPLETE"):
        DeterministicEpisodeSampler(catalog, 1)


@pytest.mark.parametrize("operation", ["support", "episode", "sample"])
@pytest.mark.parametrize("authority", ["contract", "split", "manifest", "adapter", "processing", "inventory", "W3_descriptor"])
def test_current_authority_drift_revokes_existing_materialization_and_sampler(fixture, monkeypatch, authority, operation):
    catalog = fixture.load(); plan = catalog.plans[0]
    sampler = DeterministicEpisodeSampler(catalog, 1)
    paths = {"contract": fixture.contract_path, "split": fixture.split_path, "manifest": fixture.manifest_path,
        "adapter": fixture.root / fixture.adapter["path"], "processing": fixture.root / fixture.processing["path"],
        "inventory": fixture.root / fixture.manifest["official_inventory"]["path"]}
    if authority == "W3_descriptor":
        fixture.tasks["W3"]["cache_manifest"] = {"path": fixture.manifest_path.relative_to(fixture.root).as_posix(), "sha256": "0" * 64}
        write(fixture.root / "status/tasks.json", fixture.tasks)
    elif paths[authority].suffix == ".json":
        write(paths[authority], {})
    else:
        with paths[authority].open("a") as stream:
            stream.write("\n# SYNTHETIC_POST_ISSUANCE_CODE_DRIFT\n")
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Changed authority touched numerical cache"))
    calls = {"support": lambda: catalog.materialize_support(plan), "episode": lambda: catalog.materialize(plan),
             "sample": sampler.next_plan}
    with pytest.raises(EpisodeGateError, match="artifact_sha256|authority_changed"):
        calls[operation]()
    assert sampler.cursor == 0


def test_sampler_metadata_and_selection_are_immutable_but_resume_cursor_stays_supported(fixture):
    catalog = fixture.load(); sampler = DeterministicEpisodeSampler(catalog, 7)
    with pytest.raises(TypeError):
        sampler._eligible[sampler.people[0]] = ()
    for name, value in (("catalog", object()), ("seed", 5), ("people", ()), ("_eligible", {}), ("selection_sha256", "0" * 64)):
        with pytest.raises(AttributeError, match="metadata_is_immutable"):
            setattr(sampler, name, value)
    sampler.cursor = 4
    state = sampler.state_dict()
    restored = DeterministicEpisodeSampler(catalog, 7)
    restored.load_state_dict(state)
    assert restored.cursor == 4
    assert restored.next_plan() == sampler.next_plan()


@pytest.mark.parametrize("stage,field", [("W1", "input_contract_sha256"),
                                         ("W3", "input_contract_sha256"), ("W3", "split_sha256")])
@pytest.mark.parametrize("change", ["missing", "changed"])
def test_current_task_scientific_bindings_cannot_contradict_catalog_seal(fixture, monkeypatch, stage, field, change):
    catalog = fixture.load(); plan = catalog.plans[0]
    sampler = DeterministicEpisodeSampler(catalog, 1)
    if change == "missing":
        del fixture.tasks[stage][field]
    else:
        fixture.tasks[stage][field] = "0" * 64
    write(fixture.root / "status/tasks.json", fixture.tasks)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Contradictory scientific binding touched numerical cache"))
    for operation in (lambda: catalog.materialize_support(plan), lambda: catalog.materialize(plan),
                      lambda: catalog.plan(plan.participant_id, plan.session_id), sampler.next_plan, fixture.load):
        with pytest.raises(EpisodeGateError, match="scientific_binding_mismatch"):
            operation()
    assert sampler.cursor == 0


def authorize_synthetic_internal_catalog(fixture):
    """Rebalance unaccessed test registry slots so only 11 internal payloads load.

    All identities remain a complete 598-object, 62-person software double. The
    numerical arrays and permission receipt below are entirely synthetic.
    """
    inventory_path = fixture.root / fixture.manifest["official_inventory"]["path"]
    inventory = json.loads(inventory_path.read_text())
    inventory_by_name = {entry["name"]: entry for entry in inventory["files"]}
    for session in fixture.sessions:
        if session["partition"] == "internal_holdout" and session["session_id"] != "1":
            original = inventory_by_name[session["source_record_id"]]
            person, ordinal = fixture.train[0], int(session["session_id"]) + 1
            name = f"{person}_Session_{ordinal}.mat"
            original.update(name=name, provider_subject=person, session_ordinal=ordinal)
            session.update(participant_id=person, session_id=str(ordinal), partition="meta_train", source_record_id=name)
        elif session["partition"] == "internal_holdout":
            events = fixture.root / "cache/stieger/internal_holdout" / session["participant_id"] / "1/events.json"
            write(events, fixture.rows(session))
            session["events"] = descriptor(fixture.root, events)
    write(inventory_path, inventory)
    fixture.manifest["official_inventory"] = descriptor(fixture.root, inventory_path)
    permission_path = fixture.root / "manifests/synthetic_internal_access.json"
    write(permission_path, {"schema_version": "paper3-internal-check-access-v1",
        "status": "AUTHORIZED_ONCE_DEVELOPMENT_CHECK", "role": "internal_holdout",
        "split_sha256": fixture.split["split_sha256"], "input_contract_sha256": fixture.contract_sha,
        "labels_after_predictions_only": True, "training_and_selection_allowed": False,
        "selected_configuration_sha256": "c" * 64, "software_fixture_notes": ["not real data", "no internal evaluation"]})
    permission = descriptor(fixture.root, permission_path)
    fixture.tasks["W3"]["internal_access_receipt"] = permission
    fixture.refresh()
    return fixture.load(role="internal_holdout", internal_access_receipt=permission)


def test_sealed_internal_support_rechecks_permission_and_keeps_query_labels_locked(fixture, monkeypatch):
    catalog = authorize_synthetic_internal_catalog(fixture); plan = catalog.plans[0]
    support = catalog.materialize_support(plan)
    assert support.role == "calibration_prefix" and len(support.acquired_event_uids) == 20
    episode = catalog.materialize(plan)
    assert torch.equal(support.x, episode.support.x) and torch.equal(support.y, episode.support.y)
    with pytest.raises(PermissionError, match="internal_labels_require_committed_predictions"):
        episode.evaluation_labels.labels(purpose="internal_development_check")
    del fixture.tasks["W3"]["internal_access_receipt"]
    write(fixture.root / "status/tasks.json", fixture.tasks)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Revoked internal permission touched numerical cache"))
    for operation in (lambda: catalog.materialize_support(plan), lambda: catalog.materialize(plan)):
        with pytest.raises(PermissionError, match="explicit_post_selection_access_receipt"):
            operation()


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


@pytest.mark.parametrize("starts", [(1, 26, 51), (1, 26, 300), (76, 151, 300)])
def test_complete_runs_keep_contiguous_session_global_provider_numbers(fixture, starts):
    recorded = {}
    def numbered(rows):
        for row in rows:
            run = row["physical_run_id"]
            row["provider_trialnumber"] = starts[run - 1] + row["original_trial_index"] % 25
            recorded[row["event_uid"]] = (row["original_trial_index"], row["provider_trialnumber"], run)
    fixture.mutate_rows(fixture.train[0], numbered)
    catalog = fixture.load(); plan = catalog.plan(fixture.train[0], "1")
    assert len(plan.acquired_event_uids) == 20 and len(plan.support_event_uids) == 19
    assert len(plan.query_event_uids) == 50 and plan.incomplete_future_run_uids == ()
    audit = plan.to_dict()["temporal_audit"]["original_event_audit"]
    assert {row["event_uid"]: (row["original_trial_index"], row["provider_trialnumber"], row["physical_run_id"])
            for row in audit} == recorded
    assert all(row["physical_run_complete_verified"] is True for row in audit)
    assert not (set(plan.query_event_uids) & set(plan.acquired_event_uids + plan.unused_boundary_run_uids))


@pytest.mark.parametrize("violation", ["gap", "duplicate", "descending", "zero", "noninteger", "only_24"])
def test_complete_run_claim_requires_25_actual_positive_contiguous_original_records(fixture, violation):
    def invalid(rows):
        if violation == "gap": rows[40]["provider_trialnumber"] += 2
        elif violation == "duplicate": rows[40]["provider_trialnumber"] = rows[39]["provider_trialnumber"]
        elif violation == "descending": rows[40]["provider_trialnumber"] = rows[39]["provider_trialnumber"] - 1
        elif violation == "zero": rows[25]["provider_trialnumber"] = 0
        elif violation == "noninteger": rows[25]["provider_trialnumber"] = 26.0
        else:
            rows.pop(49)
            for cache_index, row in enumerate(row for row in rows if row["valid"]):
                row["cache_index"] = cache_index
    fixture.mutate_rows(fixture.train[0], invalid)
    with pytest.raises(EpisodeGateError, match="whole_run_completion|all_25_original"):
        fixture.load()


def test_actual_24_record_run_stays_incomplete_and_later_300_to_324_run_is_usable(fixture):
    session = fixture.session(); path = fixture.root / session["events"]["path"]
    rows = json.loads(path.read_text())
    removed_uid = rows[49]["event_uid"]
    rows.pop(49)
    signals = []
    for row in rows:
        if row["physical_run_id"] == 2:
            row["physical_run_complete_verified"] = False
        elif row["physical_run_id"] == 3:
            row["provider_trialnumber"] = 300 + row["original_trial_index"] - 50
        if row["valid"]:
            row["cache_index"] = len(signals)
            signals.append(fixture.values[row["original_trial_index"]])
    x_path = path.with_name("X_with_actual_missing_record.npy")
    np.save(x_path, np.stack(signals), allow_pickle=False)
    session.update(original_lr_trials=len(rows), valid_trials=len(signals), X=descriptor(fixture.root, x_path))
    write(path, rows); session["events"] = descriptor(fixture.root, path); fixture.refresh()
    catalog = fixture.load(); plan = catalog.plan(fixture.train[0], "1")
    audit = plan.to_dict()["temporal_audit"]["original_event_audit"]
    assert len(audit) == 74 and removed_uid not in {row["event_uid"] for row in audit}
    assert len(plan.incomplete_future_run_uids) == 24 and len(plan.query_event_uids) == 25
    assert [row["provider_trialnumber"] for row in audit if row["physical_run_id"] == 3] == list(range(300, 325))
    assert {row["original_trial_index"] for row in audit} == set(range(75)) - {49}
    assert all(row["physical_run_complete_verified"] is False for row in audit if row["physical_run_id"] == 2)
    assert not (set(plan.incomplete_future_run_uids) & set(plan.query_event_uids))
    assert len(plan.acquired_event_uids) == 20 and len(plan.support_event_uids) == 19 and plan.query_eligible


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
    support = catalog.materialize_support(plan)
    assert torch.equal(support.x, episode.support.x) and torch.equal(support.y, episode.support.y)
    assert len(support.acquired_event_uids) == 20 and len(support.event_uids) == 9


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
    with pytest.raises(EpisodeGateError, match="not_evaluable"):
        catalog.materialize_support(catalog.plan(person, "1"))
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
