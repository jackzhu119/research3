"""Strict descriptor compatibility; synthetic fixtures are not EEG evidence."""
import hashlib
from types import MappingProxyType

import numpy as np
import pytest

from p3work import episodes
from test_episodes import EpisodeFixture, write


@pytest.fixture
def artifact(tmp_path):
    path = tmp_path / "cache/fixture.json"
    path.parent.mkdir(parents=True)
    raw = b'{"synthetic_fixture": true}\n'
    path.write_bytes(raw)
    return tmp_path, path, {"path": path.relative_to(tmp_path).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest()}, len(raw)


@pytest.mark.parametrize("triplet", [False, True])
@pytest.mark.parametrize("immutable", [False, True])
def test_exact_pair_and_triplet_mappings_keep_full_SHA_binding(artifact, triplet, immutable):
    root, path, descriptor, size = artifact
    if triplet:
        descriptor["bytes"] = size
    if immutable:
        descriptor = MappingProxyType(descriptor)
    assert episodes._artifact(root, descriptor, suffix=".json") == path


def test_zero_bytes_is_valid_only_for_an_actual_empty_file(tmp_path):
    path = tmp_path / "empty.json"
    path.write_bytes(b"")
    descriptor = {"path": path.name, "sha256": hashlib.sha256(b"").hexdigest(), "bytes": 0}
    assert episodes._artifact(tmp_path, descriptor) == path
    path.write_bytes(b"x")
    with pytest.raises(episodes.EpisodeGateError, match="sha256_or_type"):
        episodes._artifact(tmp_path, descriptor)


@pytest.mark.parametrize("size", [True, False, -1, 1.0, "27", None, [], {}, np.int64(27)])
def test_invalid_byte_type_or_negative_size_is_rejected_before_artifact_IO(artifact, monkeypatch, size):
    root, _, descriptor, _ = artifact
    descriptor["bytes"] = size
    monkeypatch.setattr(episodes, "_local", lambda *args, **kwargs: pytest.fail("invalid bytes must not access a path"))
    with pytest.raises(episodes.EpisodeGateError, match="nonnegative_integer_bytes"):
        episodes._artifact(root, descriptor)


@pytest.mark.parametrize("delta", [-1, 1])
def test_wrong_byte_count_is_not_ignored_even_when_SHA_matches(artifact, monkeypatch, delta):
    root, _, descriptor, size = artifact
    descriptor["bytes"] = size + delta
    monkeypatch.setattr(episodes, "file_digest", lambda *args: pytest.fail("size mismatch must fail before hashing"))
    with pytest.raises(episodes.EpisodeGateError, match="sha256_or_type"):
        episodes._artifact(root, descriptor)


@pytest.mark.parametrize("triplet", [False, True])
def test_SHA_mismatch_and_suffix_mismatch_remain_closed(artifact, triplet):
    root, _, descriptor, size = artifact
    if triplet:
        descriptor["bytes"] = size
    bad = {**descriptor, "sha256": "f"*64}
    with pytest.raises(episodes.EpisodeGateError, match="sha256_or_type"):
        episodes._artifact(root, bad)
    with pytest.raises(episodes.EpisodeGateError, match="sha256_or_type"):
        episodes._artifact(root, descriptor, suffix=".npy")


@pytest.mark.parametrize("change", ["unknown", "bytes_and_unknown", "missing_path", "missing_sha", "only_bytes", "list", "null"])
def test_only_the_two_exact_descriptor_key_sets_are_accepted(artifact, change):
    root, _, descriptor, size = artifact
    if change == "unknown": descriptor["role"] = "meta_train"
    elif change == "bytes_and_unknown": descriptor.update(bytes=size, metadata={})
    elif change == "missing_path": descriptor.pop("path")
    elif change == "missing_sha": descriptor.pop("sha256")
    elif change == "only_bytes": descriptor = {"bytes": size}
    elif change == "list": descriptor = [descriptor]
    else: descriptor = None
    with pytest.raises(episodes.EpisodeGateError, match="path_and_sha256"):
        episodes._artifact(root, descriptor)


@pytest.mark.parametrize("relative", ["../outside.json", "/tmp/outside.json", "a/./b.json", "a/../b.json",
    "a\\b.json", "cache/.env.secret", "credentials/file.json", "legacy_baseline_v1/file.json", "cache/old.npz"])
def test_triplets_preserve_all_existing_path_and_legacy_restrictions(artifact, monkeypatch, relative):
    root, _, descriptor, size = artifact
    descriptor.update(path=relative, bytes=size)
    monkeypatch.setattr(episodes, "file_digest", lambda *args: pytest.fail("unsafe paths must never be opened"))
    with pytest.raises(episodes.EpisodeGateError, match="unsafe_or_legacy"):
        episodes._artifact(root, descriptor)


def test_triplet_symlink_is_refused_without_opening_target(artifact, monkeypatch):
    root, path, descriptor, size = artifact
    linked = root / "cache/link.json"
    linked.symlink_to(path)
    descriptor.update(path=linked.relative_to(root).as_posix(), bytes=size)
    monkeypatch.setattr(episodes, "file_digest", lambda *args: pytest.fail("symlink target must not be opened"))
    with pytest.raises(episodes.EpisodeGateError, match="symlink_artifact"):
        episodes._artifact(root, descriptor)


def convert_fixture_to_producer_triplets(fixture):
    def add_bytes(value):
        if isinstance(value, dict):
            if set(value) == {"path", "sha256"}:
                path = fixture.root / value["path"]
                # Internal payloads intentionally do not exist in this fixture.
                # Their invalid size is harmless only while the role gate holds.
                value["bytes"] = path.stat().st_size if path.is_file() else 123456
            else:
                for item in value.values():
                    add_bytes(item)
        elif isinstance(value, list):
            for item in value:
                add_bytes(item)
    add_bytes(fixture.manifest)
    fixture.refresh()
    authority = fixture.tasks["W3"]["cache_manifest"]
    authority["bytes"] = fixture.manifest_path.stat().st_size
    write(fixture.root / "status/tasks.json", fixture.tasks)


@pytest.mark.parametrize("role", ["meta_train", "meta_validation"])
def test_canonical_loader_interface_accepts_producer_triplets_without_changing_episode_plans(tmp_path, role):
    fixture = EpisodeFixture(tmp_path)
    before = fixture.load(role)
    expected = [(p.participant_id, p.session_id, p.role, p.acquired_event_uids, p.support_event_uids,
        p.query_event_uids, p.unused_boundary_run_uids, p.query_eligible, p.query_rule) for p in before.plans]
    convert_fixture_to_producer_triplets(fixture)
    catalog = fixture.load(role)
    actual = [(p.participant_id, p.session_id, p.role, p.acquired_event_uids, p.support_event_uids,
        p.query_event_uids, p.unused_boundary_run_uids, p.query_eligible, p.query_rule) for p in catalog.plans]
    assert actual == expected
    assert catalog.partition_people == before.partition_people
    assert catalog.input_contract_sha256 == before.input_contract_sha256
    assert all(len(plan.acquired_event_uids) == 20 for plan in catalog.plans)
    assert all(set(plan.query_event_uids).isdisjoint(plan.acquired_event_uids) for plan in catalog.plans)
    # Catalogs are issued with metadata plans only; no support/query is materialized.
    assert catalog.plan(catalog.plans[0].participant_id, catalog.plans[0].session_id) is catalog.plans[0]


@pytest.mark.parametrize("bad_field", ["bytes", "unknown"])
def test_bad_W3_authority_triplet_cannot_issue_a_catalog(tmp_path, bad_field):
    fixture = EpisodeFixture(tmp_path)
    convert_fixture_to_producer_triplets(fixture)
    authority = fixture.tasks["W3"]["cache_manifest"]
    if bad_field == "bytes": authority["bytes"] += 1
    else: authority["extra"] = "not allowed"
    write(fixture.root / "status/tasks.json", fixture.tasks)
    with pytest.raises(episodes.EpisodeGateError):
        fixture.load("meta_train")


def test_triplet_catalog_seal_rechecks_size_and_keeps_internal_access_closed(tmp_path):
    fixture = EpisodeFixture(tmp_path)
    convert_fixture_to_producer_triplets(fixture)
    with pytest.raises(PermissionError, match="internal_holdout_requires_explicit"):
        fixture.load("internal_holdout")
    catalog = fixture.load("meta_train")
    # Preserve the SHA field while changing the bound byte assertion. The seal
    # rejects stage authority drift before exposing any later capability.
    fixture.tasks["W3"]["cache_manifest"]["bytes"] += 1
    write(fixture.root / "status/tasks.json", fixture.tasks)
    with pytest.raises(episodes.EpisodeGateError, match="authority_changed"):
        catalog.plan(catalog.plans[0].participant_id, catalog.plans[0].session_id)
