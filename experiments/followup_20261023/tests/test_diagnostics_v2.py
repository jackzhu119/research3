"""Synthetic stress is paired across methods, not fitted on future query data."""
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from paper3v2.algorithm import Config, REFERENCE_VARIANTS
from paper3v2.diagnostics import SCENARIOS, perturb_episode
from p3work.spatial_geometry import SpatialGeometry
from test_algorithm import fixture


def plan(**extras):
    return SimpleNamespace(participant_id="S037", session_id="2", **extras)


def test_clean_episode_is_unchanged_and_all_six_scenarios_are_declared():
    assert set(SCENARIOS) == {"label_noise10", "label_noise20", "gain10", "gain25",
                              "rotation10", "rotation25"}
    _, _, geometry, episode, _ = fixture()
    clean, metadata = perturb_episode(episode, geometry, plan(), "clean")
    assert clean is episode
    assert metadata == {"synthetic_stress": False}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_stress_preserves_input_tensors_and_backing_cache(tmp_path, scenario):
    _, _, geometry, original, _ = fixture()
    path = tmp_path / "cached_signals.npy"
    np.save(path, np.concatenate((original.support.x.numpy(), original.query_x.numpy())))
    array = np.load(path, mmap_mode="r+")
    tensor = torch.from_numpy(array)
    episode = SimpleNamespace(**vars(original))
    episode.support = replace(original.support, x=tensor[:20])
    episode.query_x = tensor[20:]
    before_bytes = sha256(path.read_bytes()).hexdigest()
    before_x = tensor.clone()
    before_y, before_qy = episode.support.y.clone(), episode.query_y.clone()
    modified, metadata = perturb_episode(episode, geometry, plan(), scenario)
    array.flush()
    assert sha256(path.read_bytes()).hexdigest() == before_bytes
    assert torch.equal(tensor, before_x)
    assert torch.equal(episode.support.y, before_y)
    assert torch.equal(episode.query_y, before_qy)
    assert modified.query_y is episode.query_y
    assert modified.run_ids == episode.run_ids
    assert metadata["synthetic_stress"] is True
    if scenario.startswith("label_noise"):
        expected_count = 2 if scenario.endswith("10") else 4
        flips = (modified.support.y != before_y).nonzero().flatten().tolist()
        assert len(flips) == expected_count == metadata["label_flips_actual"]
        assert set(metadata["affected_support_ids"]) == {episode.support.event_uids[i] for i in flips}
        assert modified.query_x is episode.query_x
        assert modified.support.x is episode.support.x
        assert metadata["query_changed"] is False
        assert modified.support.y.data_ptr() != episode.support.y.data_ptr()
    else:
        assert modified.support is episode.support
        assert not torch.equal(modified.query_x, episode.query_x)
        assert metadata["support_changed"] is False
        assert metadata["query_changed"] is True
        assert float(modified.query_x.mean(1).abs().max()) < 1e-12


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_stress_is_repeatable_across_methods_seeds_and_source_references(scenario):
    _, _, geometry, episode, _ = fixture()
    first, first_metadata = perturb_episode(episode, geometry, plan(method="P", seed=20261018), scenario)
    # Source covariance differs across pipeline seeds. It must not define the
    # physical perturbation; otherwise algorithms would face different inputs.
    reference = torch.diag(torch.arange(1, 11, dtype=torch.float64)) / 55
    different_geometry = SpatialGeometry.from_reference(reference)
    torch.manual_seed(1001)
    torch.randn(300)
    second, second_metadata = perturb_episode(episode, different_geometry,
                                             plan(method="dual", seed=20261020), scenario)
    assert first_metadata == second_metadata
    assert torch.equal(first.query_x, second.query_x)
    assert torch.equal(first.support.x, second.support.x)
    assert torch.equal(first.support.y, second.support.y)


def test_label_noise_uses_nested_flip_indices_across_severity():
    _, _, geometry, episode, _ = fixture()
    mild, m10 = perturb_episode(episode, geometry, plan(), "label_noise10")
    strong, m20 = perturb_episode(episode, geometry, plan(), "label_noise20")
    assert m10["direction_seed"] == m20["direction_seed"]
    assert set(m10["affected_support_ids"]) < set(m20["affected_support_ids"])
    assert (mild.support.y != episode.support.y).sum() == 2
    assert (strong.support.y != episode.support.y).sum() == 4


@pytest.mark.parametrize("kind", ("gain", "rotation"))
def test_spatial_strengths_share_direction_and_form_the_same_matrix_flow(kind):
    _, _, geometry, episode, _ = fixture()
    mild, m10 = perturb_episode(episode, geometry, plan(), kind + "10")
    strong, m25 = perturb_episode(episode, geometry, plan(), kind + "25")
    assert m10["direction_seed"] == m25["direction_seed"]
    assert m10["log_generator_frobenius_norm"] == .1
    assert m25["log_generator_frobenius_norm"] == .25
    assert m10["units"] == "CAR_subspace_log_transform_norm_not_percent_gain"
    # Recover the actual CAR-space operator from its action on the query, then
    # compare its flow: two .25 applications equal five .10 applications.
    def flatten(x):
        return (geometry.q @ x).transpose(0, 1).reshape(10, -1)
    inverse = torch.linalg.pinv(flatten(episode.query_x))
    a10, a25 = flatten(mild.query_x) @ inverse, flatten(strong.query_x) @ inverse
    torch.testing.assert_close(torch.linalg.matrix_power(a25, 2), torch.linalg.matrix_power(a10, 5),
                               atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(torch.linalg.det(a25), torch.ones((), dtype=torch.float64),
                               atol=1e-11, rtol=1e-11)
    if kind == "gain":
        torch.testing.assert_close(a25, a25.T, atol=1e-12, rtol=1e-12)
        assert bool((torch.linalg.eigvalsh(a25) > 0).all())
    else:
        torch.testing.assert_close(a25.T @ a25, torch.eye(10, dtype=torch.float64),
                                   atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("variant", REFERENCE_VARIANTS)
def test_reference_variants_can_use_worker_config(variant):
    assert Config(variant=variant).variant == variant
