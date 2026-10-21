"""Synthetic metadata fixtures only; no empirical prediction, fit, or network."""
import copy
import hashlib
from unittest.mock import patch

import numpy as np
import pytest

from p3work import episodes
from p3work import prediction_store as store
from p3work.contracts import file_digest
from tests.test_episode_artifact_descriptor_compatibility import convert_fixture_to_producer_triplets
from tests.test_prediction_store import fixture as prediction_fixture


@pytest.fixture
def artifact(tmp_path):
    path = tmp_path / "cache/fixture.json"
    path.parent.mkdir(parents=True)
    body = b'{"synthetic_fixture":true}\n'
    path.write_bytes(body)
    return tmp_path, path, {"path": path.relative_to(tmp_path).as_posix(),
        "sha256": hashlib.sha256(body).hexdigest()}, len(body)


@pytest.mark.parametrize("triplet", [False, True])
def test_reference_accepts_exact_pair_and_triplet_with_full_SHA(artifact, triplet):
    root, path, descriptor, size = artifact
    if triplet:
        descriptor["bytes"] = size
    assert store._reference(root, descriptor, suffix=".json") == path


def test_zero_byte_reference_requires_actual_empty_file(tmp_path):
    path = tmp_path / "empty.json"
    path.write_bytes(b"")
    ref = {"path": path.name, "sha256": hashlib.sha256(b"").hexdigest(), "bytes": 0}
    assert store._reference(tmp_path, ref) == path
    path.write_bytes(b"x")
    with pytest.raises(store.PredictionStoreError, match="full_sha256_or_type"):
        store._reference(tmp_path, ref)


@pytest.mark.parametrize("size", [True, False, -1, 1.0, "27", None, [], {}, np.int64(27)])
def test_reference_rejects_bad_bytes_before_path_IO(artifact, monkeypatch, size):
    root, _, ref, _ = artifact
    ref["bytes"] = size
    monkeypatch.setattr(store, "_local", lambda *args, **kwargs: pytest.fail("bad bytes must not access path"))
    with pytest.raises(store.PredictionStoreError, match="nonnegative_integer_bytes"):
        store._reference(root, ref)


@pytest.mark.parametrize("delta", [-1, 1])
def test_reference_rejects_wrong_actual_size_before_digest(artifact, monkeypatch, delta):
    root, _, ref, size = artifact
    ref["bytes"] = size + delta
    monkeypatch.setattr(store, "file_digest", lambda *args: pytest.fail("wrong size must fail before hashing"))
    with pytest.raises(store.PredictionStoreError, match="full_sha256_or_type"):
        store._reference(root, ref)


@pytest.mark.parametrize("change", ["extra", "bytes_extra", "missing_path", "missing_sha", "list", "null"])
def test_reference_rejects_unknown_or_missing_descriptor_keys(artifact, change):
    root, _, ref, size = artifact
    if change == "extra": ref["extra"] = None
    elif change == "bytes_extra": ref.update(bytes=size, extra=None)
    elif change == "missing_path": ref.pop("path")
    elif change == "missing_sha": ref.pop("sha256")
    elif change == "list": ref = [ref]
    else: ref = None
    with pytest.raises(store.PredictionStoreError, match="explicit_path_sha256_reference"):
        store._reference(root, ref)


@pytest.mark.parametrize("relative", ["../outside.json", "/tmp/outside.json", "a/./b.json", "a/../b.json",
    "a\\b.json", "cache/.env.secret", "credentials/file.json", "legacy_baseline_v1/file.json", "cache/old.npz"])
def test_reference_triplets_keep_path_restrictions(artifact, monkeypatch, relative):
    root, _, ref, size = artifact
    ref.update(path=relative, bytes=size)
    monkeypatch.setattr(store, "file_digest", lambda *args: pytest.fail("unsafe path must not be hashed"))
    with pytest.raises(store.PredictionStoreError, match="unsafe_credential_legacy_or_external"):
        store._reference(root, ref)


def test_reference_triplet_still_rejects_symlinks_bad_SHA_and_wrong_suffix(artifact):
    root, path, ref, size = artifact
    ref["bytes"] = size
    with pytest.raises(store.PredictionStoreError, match="full_sha256_or_type"):
        store._reference(root, {**ref, "sha256": "f"*64})
    with pytest.raises(store.PredictionStoreError, match="full_sha256_or_type"):
        store._reference(root, ref, suffix=".npy")
    link = path.with_name("link.json")
    link.symlink_to(path)
    with pytest.raises(store.PredictionStoreError, match="symlink_artifact"):
        store._reference(root, {**ref, "path": link.relative_to(root).as_posix()})


def producer_triplet_prediction_fixture(tmp_path):
    native, _, _, refs, _, _ = prediction_fixture(tmp_path)
    convert_fixture_to_producer_triplets(native)
    refs["episode_manifest"] = copy.deepcopy(native.tasks["W3"]["cache_manifest"])
    refs["input_contract"] = copy.deepcopy(native.manifest["input_contract"])
    refs["participant_split"] = copy.deepcopy(native.manifest["participant_split"])
    for ref in refs.values():
        ref["bytes"] = (tmp_path / ref["path"]).stat().st_size
    catalog = native.load(role="meta_validation")
    return native, catalog, refs


def test_canonical_W3_triplet_flows_from_catalog_to_prediction_validation_without_materializing(tmp_path):
    with patch.object(episodes.VerifiedEpisodeCatalog, "materialize", side_effect=AssertionError("no materialization")), \
            patch.object(episodes.VerifiedEpisodeCatalog, "materialize_support", side_effect=AssertionError("no support")), \
            patch.object(episodes.EvaluationLabels, "labels", side_effect=AssertionError("no query labels")):
        native, catalog, refs = producer_triplet_prediction_fixture(tmp_path)
        plan = catalog.plans[0]
        audit = store._validate_plan(tmp_path, plan, refs)
    assert refs["episode_manifest"] == native.tasks["W3"]["cache_manifest"]
    assert set(refs["episode_manifest"]) == {"path", "sha256", "bytes"}
    assert audit["role"] == "meta_validation"
    assert audit["acquired_prefix_uids"] == list(plan.acquired_event_uids)
    assert audit["query_uids"] == list(plan.query_event_uids)
    assert len(plan.acquired_event_uids) == 20
    assert set(plan.acquired_event_uids).isdisjoint(plan.query_event_uids)
    assert not (tmp_path / "predictions").exists()


@pytest.mark.parametrize("size", [True, False, -1, 1.0, "1", None, [], {}, np.int64(1)])
def test_plan_shape_rejects_bad_bytes_without_accessing_any_reference(tmp_path, monkeypatch, size):
    _, catalog, refs = producer_triplet_prediction_fixture(tmp_path)
    refs["model"]["bytes"] = size
    monkeypatch.setattr(store, "_local", lambda *args, **kwargs: pytest.fail("shape must fail before path access"))
    with pytest.raises(store.PredictionStoreError, match="nonnegative_integer_bytes"):
        store._validate_plan(tmp_path, catalog.plans[0], refs)


@pytest.mark.parametrize("change", ["unknown_key", "pair_projection", "wrong_size", "foreign_path"])
def test_W3_triplet_is_pinned_exactly_before_any_supplied_reference_body(tmp_path, monkeypatch, change):
    _, catalog, refs = producer_triplet_prediction_fixture(tmp_path)
    if change == "unknown_key": refs["episode_manifest"]["extra"] = None
    elif change == "pair_projection": refs["episode_manifest"].pop("bytes")
    elif change == "wrong_size": refs["episode_manifest"]["bytes"] += 1
    else: refs["episode_manifest"]["path"] = "cache/stieger/does_not_exist.json"
    original_read = store._read_json
    def only_status(path):
        assert path == tmp_path / "status/tasks.json", "metadata pin must precede supplied body reads"
        return original_read(path)
    monkeypatch.setattr(store, "_read_json", only_status)
    monkeypatch.setattr(store, "_reference", lambda *args, **kwargs: pytest.fail("unpinned supplied reference must not open"))
    with pytest.raises(store.PredictionStoreError, match="explicit_path_sha256_reference|W3_complete_receipt"):
        store._validate_plan(tmp_path, catalog.plans[0], refs)


def test_bound_triplet_reference_size_must_match_even_after_metadata_pin(tmp_path):
    _, catalog, refs = producer_triplet_prediction_fixture(tmp_path)
    refs["model"]["bytes"] += 1
    assert refs["model"]["sha256"] == file_digest(tmp_path / refs["model"]["path"])
    with pytest.raises(store.PredictionStoreError, match="full_sha256_or_type"):
        store._validate_plan(tmp_path, catalog.plans[0], refs)
