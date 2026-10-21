"""Pure CPU float64 mathematics; no learned weights, EEG, GPU or R2."""
import math

import pytest
import torch

from p3work.diagnostics import (coefficient_diagnostics, dictionary_subspace_diagnostics,
    exp_transform_diagnostics, generator_energy, parameter_count_table, subspace_principal_angles)
from p3work.spatial_geometry import (generators, helmert_rows, initialize_dictionary,
                                   positive_qr, traceless_basis)


def dictionary(r=4, seed=7, d=10, kind="full"):
    return generators(traceless_basis(d, kind=kind), initialize_dictionary(r, seed, d=d, kind=kind))


def test_whole_dictionary_energy_is_orthogonal_decomposition_and_gauge_invariant():
    g = dictionary()
    rotation = positive_qr(torch.arange(16, dtype=torch.float64).reshape(4, 4) + torch.eye(4))
    mixed = torch.einsum("ij,ikl->jkl", rotation, g)
    first, second = generator_energy(g), generator_energy(mixed)
    assert first["total_energy"] == pytest.approx(4)
    assert first["symmetric_fraction"] + first["skew_fraction"] == pytest.approx(1)
    assert first["symmetric_energy"] == pytest.approx(second["symmetric_energy"], abs=1e-14)
    assert first["skew_energy"] == pytest.approx(second["skew_energy"], abs=1e-14)
    assert first["real_mechanism_conclusion"] is False


@pytest.mark.parametrize("kind,key", [("symmetric", "symmetric_fraction"), ("skew", "skew_fraction")])
def test_restricted_dictionary_has_expected_whole_energy(kind, key):
    result = generator_energy(dictionary(kind=kind))
    assert result[key] == pytest.approx(1, abs=1e-14)


def test_principal_angles_ignore_sign_permutation_and_orthogonal_column_rotation():
    g = dictionary()
    rotation = positive_qr(torch.arange(16, dtype=torch.float64).reshape(4, 4) + torch.eye(4))
    mixed = torch.einsum("ij,ikl->jkl", rotation, g).flip(0)
    mixed[0] *= -1
    result = dictionary_subspace_diagnostics(g, mixed)
    assert max(result["principal_angles_radians"]) < 2e-14
    assert result["projector_frobenius_distance"] < 2e-14
    assert result["squared_subspace_overlap"] == pytest.approx(4)


def test_known_principal_angle_and_unequal_dimension_projectors():
    angle = .37
    left = torch.eye(4, dtype=torch.float64)[:, :2]
    right = torch.stack((left[:, 0], torch.tensor([0, math.cos(angle), math.sin(angle), 0], dtype=torch.float64)), dim=1)
    measured = subspace_principal_angles(left, right)
    assert measured["principal_angles_radians"] == pytest.approx([0, angle], abs=1e-14)
    nested = subspace_principal_angles(left[:, :1], left)
    assert nested["principal_angles_radians"] == pytest.approx([0], abs=1e-14)
    assert nested["unmatched_dimensions"] == 1
    assert nested["projector_frobenius_distance"] == pytest.approx(1)
    orthogonal = subspace_principal_angles(left, torch.eye(4, dtype=torch.float64)[:, 2:])
    assert orthogonal["principal_angles_radians"] == pytest.approx([math.pi / 2] * 2)


def test_symmetric_pca_and_unrestricted_dictionary_share_matrix_ambient_frame():
    symmetric = traceless_basis(10, kind="symmetric").T.reshape(54, 10, 10)[:2]
    skew = traceless_basis(10, kind="skew").T.reshape(45, 10, 10)[:2]
    result = dictionary_subspace_diagnostics(symmetric, skew)
    assert result["ambient_dimension"] == 100
    assert result["principal_angles_radians"] == pytest.approx([math.pi / 2] * 2)


def test_actual_M_norm_equals_theta_and_boundary_rate_has_explicit_roundoff_definition():
    g = dictionary(r=2)
    theta = torch.tensor([[0., 0.], [.15, 0.], [.03, .04]], dtype=torch.float64)
    result = coefficient_diagnostics(g, theta, .15)
    assert result["M_frobenius_norms"] == pytest.approx([0, .15, .05], abs=1e-14)
    assert result["theta_l2_norms"] == pytest.approx(result["M_frobenius_norms"], abs=1e-14)
    assert result["roundoff_boundary_hits"] == [False, True, False]
    assert result["roundoff_boundary_hit_rate"] == pytest.approx(1 / 3)
    assert result["radius_minus_theta_l2"] == pytest.approx([.15, 0, .1])
    assert result["boundary_comparison_atol"] == torch.finfo(torch.float64).eps * 2
    assert all(op <= frob + 1e-14 for op, frob in zip(result["M_operator_2_norms"], result["M_frobenius_norms"]))


def test_theta_inverse_rotation_keeps_actual_M_and_norms_unchanged():
    g = dictionary(r=2)
    rotation = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
    mixed = torch.einsum("ij,ikl->jkl", rotation, g)
    theta = torch.tensor([.04, .06], dtype=torch.float64)
    first = coefficient_diagnostics(g, theta, .15)
    second = coefficient_diagnostics(mixed, rotation.T @ theta, .15)
    torch.testing.assert_close(first["M_matrices"], second["M_matrices"], atol=1e-16, rtol=1e-14)


def test_one_coefficient_can_produce_full_rank_matrix_update_and_exact_parameter_counts():
    diagonal = torch.tensor([1., -1.] * 5, dtype=torch.float64).diag() / math.sqrt(10)
    result = exp_transform_diagnostics(diagonal[None], torch.tensor([.1], dtype=torch.float64), .15,
                                       torch.eye(10, dtype=torch.float64) / 10)
    assert result["whitened_update_numerical_rank"]["numerical_rank"] == 10
    assert result["coefficients"]["coefficient_count_r"] == 1
    counts = parameter_count_table(128, r=4, k=4)
    assert counts["generator"]["target_total_parameters"] == 133
    assert counts["free_or_meta_lora"]["target_total_parameters"] == 209
    assert counts["equal_target_parameter_count"] is False
    assert counts["generator"]["dictionary_tensor_entries"] == 396
    assert counts["generator"]["Stiefel_intrinsic_dimension"] == 386
    assert counts["generator"]["Grassmann_subspace_dimension"] == 380
    assert counts["generator"]["orthogonal_basis_gauge_dimension"] == 6
    assert counts["free_or_meta_lora"]["rank_exactly_k_matrix_manifold_dimension"] == 64


def test_white_space_rotation_bound_does_not_transfer_to_sensor_euclidean_condition():
    g = torch.zeros(1, 10, 10, dtype=torch.float64)
    g[0, 0, 1], g[0, 1, 0] = 1 / math.sqrt(2), -1 / math.sqrt(2)
    reference = torch.diag(torch.tensor([.955] + [.005] * 9, dtype=torch.float64))
    result = exp_transform_diagnostics(g, torch.tensor([.35], dtype=torch.float64), .35, reference)
    assert result["whitened_condition_number"] == pytest.approx(1, abs=1e-14)
    assert result["whitened_determinant"] == pytest.approx(1, abs=1e-14)
    assert result["sensor_CAR_restriction_condition_number"] > math.exp(.7)
    assert result["sensor_full_condition_number"] is None
    assert result["sensor_structural_determinant"] == 0
    assert result["sensor_structural_rank"] == 10
    assert result["sensor_numerical_rank"]["numerical_rank"] == 10
    assert result["whitened_singular_value_bound_applies_to_sensor_euclidean_coordinates"] is False
    assert result["post_RMS_volume_preservation_claim"] is False


def test_zero_theta_is_CAR_not_invertible_identity_on_all_sensor_coordinates():
    q = helmert_rows()
    result = exp_transform_diagnostics(dictionary(r=2), torch.zeros(2, dtype=torch.float64), .15,
                                       torch.eye(10, dtype=torch.float64) / 10)
    torch.testing.assert_close(result["A_sensor"], q.T @ q, atol=2e-15, rtol=0)
    assert result["M_numerical_rank"]["numerical_rank"] == 0
    assert result["whitened_update_numerical_rank"]["numerical_rank"] == 0
    assert result["sensor_common_reference_annihilation_error"] < 1e-14
    nonisotropic = torch.diag(torch.arange(1, 11, dtype=torch.float64) / 55)
    measured = exp_transform_diagnostics(dictionary(r=2), torch.zeros(2, dtype=torch.float64), .15, nonisotropic)
    assert measured["sensor_change_from_CAR_numerical_rank"]["numerical_rank"] == 0
    assert measured["sensor_change_from_CAR_numerical_rank"]["subtraction_reference_scale"] >= 1


@pytest.mark.parametrize("violation", ["float32", "nonfinite", "trace", "nonorthogonal", "empty"])
def test_invalid_generator_inputs_are_rejected_without_repair(violation):
    g = dictionary()
    if violation == "float32": g = g.float()
    elif violation == "nonfinite": g[0, 0, 0] = float("nan")
    elif violation == "trace": g[0] += torch.eye(10, dtype=torch.float64)
    elif violation == "nonorthogonal": g[1] = g[0]
    else: g = g[:0]
    with pytest.raises(ValueError): generator_energy(g)


@pytest.mark.parametrize("violation", ["rho", "out_of_ball", "theta_shape", "reference", "Q", "basis"])
def test_invalid_radius_reference_coordinate_and_subspace_inputs_fail_closed(violation):
    g = dictionary(r=2); theta = torch.zeros(2, dtype=torch.float64); radius = .15
    reference, q = torch.eye(10, dtype=torch.float64), helmert_rows()
    if violation == "rho": radius = 0
    elif violation == "out_of_ball": theta[0] = .151
    elif violation == "theta_shape": theta = theta[:1]
    elif violation == "reference": reference[0, 0] = 0
    elif violation == "Q": q = -q
    else:
        with pytest.raises(ValueError): subspace_principal_angles(torch.ones(4, 2, dtype=torch.float64), torch.eye(4, dtype=torch.float64))
        return
    with pytest.raises(ValueError): exp_transform_diagnostics(g, theta, radius, reference, q)


def test_diagnostics_leave_global_rng_and_input_arrays_unchanged():
    g = dictionary(); theta = torch.zeros(4, dtype=torch.float64)
    before_g, before_theta, rng = g.clone(), theta.clone(), torch.get_rng_state().clone()
    exp_transform_diagnostics(g, theta, .15, torch.eye(10, dtype=torch.float64) / 10)
    dictionary_subspace_diagnostics(g, g)
    generator_energy(g)
    assert torch.equal(g, before_g) and torch.equal(theta, before_theta)
    assert torch.equal(torch.get_rng_state(), rng)
