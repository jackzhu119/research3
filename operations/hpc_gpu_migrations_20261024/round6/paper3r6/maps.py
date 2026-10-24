"""Parameter-matched charts for traceless CAR-space spatial generators.

All identities in this module concern the ten-dimensional reference-whitened
operator. Reference recoloring is a nonorthogonal similarity and does not
preserve Euclidean singular values or perturbation norms in CAR coordinates.
"""
import math
import torch

GEOMETRIES = ('single', 'left_rotation', 'right_rotation', 'symmetric', 'bch2')


def split_generator(matrix):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Expected a square generator matrix')
    return (matrix - matrix.T) / 2, (matrix + matrix.T) / 2


def matrix_map(matrix, geometry='single'):
    """Actual nonlinear operator; multiplication is written left to right.

    For column-vector signals the rightmost factor acts first. The three
    product charts use the same generator coordinates, not extra parameters.
    ``bch2`` is the second-order log approximation to exp(K) exp(S), retained
    as a mechanism probe rather than a primary fitted model.
    """
    if geometry not in GEOMETRIES:
        raise ValueError('Unknown spatial geometry: ' + str(geometry))
    if geometry == 'single':
        return torch.matrix_exp(matrix)
    k, s = split_generator(matrix)
    if geometry == 'left_rotation':
        return torch.matrix_exp(k) @ torch.matrix_exp(s)
    if geometry == 'right_rotation':
        return torch.matrix_exp(s) @ torch.matrix_exp(k)
    if geometry == 'symmetric':
        half = torch.matrix_exp(k / 2)
        return half @ torch.matrix_exp(s) @ half
    return torch.matrix_exp(matrix + (k @ s - s @ k) / 2)


def transform_diagnostics(matrix, geometry='single', rref_sqrt=None, rref_invsqrt=None):
    """Detached finite matrix diagnostics; no signal or query labels accepted."""
    with torch.no_grad():
        m = matrix.detach().double()
        k, s = split_generator(m)
        t = matrix_map(m, geometry)
        eye = torch.eye(len(m), device=m.device, dtype=m.dtype)
        values = torch.linalg.svdvals(t)
        defect = matrix_map(-m, geometry) @ t - eye
        out = {'geometry': geometry, 'generator_norm': float(m.norm()),
               'rotation_generator_norm': float(k.norm()), 'gain_generator_norm': float(s.norm()),
               'commutator_norm': float((k @ s - s @ k).norm()),
               'white_transform_deviation': float((t-eye).norm()),
               'white_relative_transform_deviation': float((t-eye).norm()/math.sqrt(len(m))),
               'white_condition': float(values.max()/values.min()),
               'white_log_singular_spread': float(values.max().log()-values.min().log()),
               'white_determinant': float(torch.linalg.det(t)),
               'inverse_consistency_defect': float(defect.norm()),
               'matrix_coordinate_scope': 'reference_whitened_CAR10',
               'ambient_generator_dimension': int(m.numel()-1)}
        if rref_sqrt is not None or rref_invsqrt is not None:
            if rref_sqrt is None or rref_invsqrt is None:
                raise ValueError('Both reference root matrices are required')
            raw = rref_sqrt.to(t) @ t @ rref_invsqrt.to(t)
            raw_values = torch.linalg.svdvals(raw)
            out.update(car_transform_deviation=float((raw-eye).norm()),
                       car_relative_transform_deviation=float((raw-eye).norm()/math.sqrt(len(m))),
                       car_condition=float(raw_values.max()/raw_values.min()))
        if geometry in ('left_rotation', 'right_rotation', 'symmetric'):
            u, _, vh = torch.linalg.svd(t)
            out['white_polar_rotation_error'] = float((u @ vh - torch.matrix_exp(k)).norm())
        return out
