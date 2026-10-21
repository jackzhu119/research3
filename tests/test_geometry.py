"""CPU synthetic numerical acceptance; these are NOT research pilot results."""
import math

import pytest
import torch

from p3work.spatial_geometry import (SourceCovarianceTrial, SpatialGeometry, generators,
    helmert_rows, initialize_dictionary, joint_trial_rms, positive_qr, project_ball,
    source_reference, stiefel_step, tensor_sha256, traceless_basis)


torch.set_num_threads(1)


def test_q_e_b_are_deterministic_orthogonal_and_g_is_traceless():
    q = helmert_rows()
    h = torch.eye(11, dtype=torch.float64) - torch.ones(11, 11, dtype=torch.float64) / 11
    torch.testing.assert_close(q @ q.T, torch.eye(10, dtype=torch.float64), atol=1e-14, rtol=0)
    torch.testing.assert_close(q.T @ q, h, atol=1e-14, rtol=0)
    assert tensor_sha256(q) == tensor_sha256(helmert_rows())
    for kind, n in (("full", 99), ("symmetric", 54), ("skew", 45)):
        e = traceless_basis(kind=kind)
        assert e.shape == (100, n)
        torch.testing.assert_close(e.T @ e, torch.eye(n, dtype=torch.float64), atol=1e-14, rtol=0)
        b = initialize_dictionary(4, 20261018, kind=kind)
        assert torch.equal(b, initialize_dictionary(4, 20261018, kind=kind))
        torch.testing.assert_close(b.T @ b, torch.eye(4, dtype=torch.float64), atol=1e-14, rtol=0)
        g = generators(e, b)
        torch.testing.assert_close(torch.diagonal(g, dim1=1, dim2=2).sum(1), torch.zeros(4, dtype=torch.float64), atol=1e-14, rtol=0)
        torch.testing.assert_close(g.flatten(1) @ g.flatten(1).T, torch.eye(4, dtype=torch.float64), atol=1e-14, rtol=0)


def test_geometry_bounds_in_whitened_coordinates_and_reference_invariance():
    rng = torch.Generator().manual_seed(13)
    z = torch.randn(10, 10, dtype=torch.float64, generator=rng)
    reference = z @ z.T + torch.eye(10, dtype=torch.float64)
    geometry = SpatialGeometry.from_reference(reference)
    b = initialize_dictionary(4, 3)
    theta = project_ball(torch.randn(4, dtype=torch.float64, generator=rng), 0.35)
    a, t, m = geometry.matrix(theta, b)
    torch.testing.assert_close(m.norm(), theta.norm(), atol=1e-14, rtol=0)
    singular = torch.linalg.svdvals(t)
    assert singular.min() >= math.exp(-0.35) - 1e-12
    assert singular.max() <= math.exp(0.35) + 1e-12
    assert singular.max() / singular.min() <= math.exp(0.7) + 1e-12
    torch.testing.assert_close(torch.det(t), torch.tensor(1.0, dtype=torch.float64), atol=1e-12, rtol=0)
    # The original-coordinate CAR operator has rank 10: do NOT assert det(A)=1.
    assert torch.linalg.matrix_rank(a) == 10
    x = torch.randn(3, 11, 17, dtype=torch.float64, generator=rng)
    offset = torch.randn(3, 1, 17, dtype=torch.float64, generator=rng)
    torch.testing.assert_close(geometry.transform(x, theta, b), geometry.transform(x + offset, theta, b), atol=1e-12, rtol=1e-12)
    y = a @ x
    torch.testing.assert_close(y.mean(1), torch.zeros(3, 17, dtype=torch.float64), atol=1e-12, rtol=0)


def test_zero_identity_and_nonzero_skew_rotation():
    geometry = SpatialGeometry.from_reference(torch.eye(10, dtype=torch.float64))
    b = initialize_dictionary(2, 5)
    rng = torch.Generator().manual_seed(4)
    x = torch.randn(4, 11, 7, dtype=torch.float64, generator=rng)
    x = x - x.mean(1, keepdim=True)
    torch.testing.assert_close(geometry.transform(x, torch.zeros(2, dtype=torch.float64), b), joint_trial_rms(x), atol=1e-14, rtol=1e-14)
    skew = SpatialGeometry.from_reference(torch.eye(10, dtype=torch.float64), kind="skew")
    _, t, _ = skew.matrix(torch.tensor([0.15], dtype=torch.float64), initialize_dictionary(1, 6, kind="skew"))
    torch.testing.assert_close(t.T @ t, torch.eye(10, dtype=torch.float64), atol=1e-13, rtol=0)
    assert (t - torch.eye(10, dtype=torch.float64)).norm() > 0.1


def test_rms_has_no_amplitude_floor_and_rejects_bad_trials():
    x = torch.arange(1, 23, dtype=torch.float64).reshape(1, 11, 2)
    torch.testing.assert_close(joint_trial_rms(x), joint_trial_rms(x * 1e-120), atol=1e-14, rtol=1e-14)
    for bad in (torch.zeros_like(x), x * float("nan"), x * float("inf")):
        with pytest.raises(ValueError):
            joint_trial_rms(bad)


def test_project_and_stiefel_step():
    zero = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    project_ball(zero, 0.15).sum().backward()
    assert torch.isfinite(zero.grad).all()
    theta = project_ball(torch.tensor([3.0, 4.0], dtype=torch.float64), 0.15)
    assert abs(float(theta.norm()) - 0.15) < 1e-14
    b = initialize_dictionary(4, 7)
    next_b = stiefel_step(b, torch.ones_like(b) * 10)
    torch.testing.assert_close(next_b.T @ next_b, torch.eye(4, dtype=torch.float64), atol=1e-14, rtol=0)


def test_source_reference_hierarchical_weighting_and_visibility():
    q = helmert_rows()
    sources = ["BNCI2014_001", "Cho2017", "Lee2019_MI", "PhysionetMI"]
    trials = []
    for dataset in sources:
        for subject, direction, repetitions in (("a", 0, 3), ("b", 1, 1)):
            v = torch.zeros(10, 2, dtype=torch.float64)
            v[direction] = 1
            trials.extend(SourceCovarianceTrial(dataset, subject, f"{dataset}/{subject}/{i}", q.T @ v)
                          for i in range(repetitions))
    ref = source_reference(trials)
    expected = torch.eye(10, dtype=torch.float64) * 0.005
    expected[0, 0] += 0.475
    expected[1, 1] += 0.475
    torch.testing.assert_close(ref.matrix, expected, atol=1e-14, rtol=0)
    torch.testing.assert_close(ref.sqrt @ ref.invsqrt, torch.eye(10, dtype=torch.float64), atol=1e-14, rtol=0)
    assert len(ref.event_uids) == 16
    assert ref.weighting == "dataset_equal/subject_equal/trial_equal"
    with pytest.raises(PermissionError):
        source_reference([SourceCovarianceTrial("Stieger2021", "a", "bad", trials[0].x)])
    with pytest.raises(PermissionError):
        source_reference([SourceCovarianceTrial(sources[0], "a", "bad", trials[0].x, role="source_validation")])
    with pytest.raises(ValueError):
        source_reference(trials + [trials[0]])
