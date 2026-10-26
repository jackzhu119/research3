"""Matched R11 mechanism-control checks; CPU synthetic inputs only."""
from dataclasses import replace
import inspect

import numpy as np
import pytest

from paper3r11 import r10_profile as a
from paper3r11.ablations import METHODS, predict_ablation


def fitted(config=None, labels=None):
    x = np.array([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]])
    y = np.array([0, 0, 1, 1]) if labels is None else np.array(labels)
    cfg = config or a.Config(amplitude=2., regularization=.7, evidence_fraction=.3,
                             include_intercept=False)
    return a.fit(x, np.zeros((4, 2)), y, cfg,
                 support_ids=('support-0', 'support-1', 'support-2', 'support-3'))


def manual_geometry(state, query):
    # Independent hand formula for the fixed linear/no-intercept fixture.
    cross = state.config.amplitude**2*(state.features@query.T)/query.shape[1]
    coefficients = np.linalg.solve(state.factor, cross)
    diagonal = state.config.amplitude**2*np.mean(query*query, axis=1)+state.config.nugget
    z = state.margins+state.factor@state.map_parameter
    p = 1/(1+np.exp(-z))
    h = (state.factor.T*(state.weights*p*(1-p)))@state.factor+state.config.regularization*np.eye(4)
    g = state.factor.T@(state.weights*(p-state.labels))+state.config.regularization*state.map_parameter
    f = float(state.weights@(np.logaddexp(0., z)-state.labels*z)
              +.5*state.config.regularization*(state.map_parameter@state.map_parameter))
    return coefficients, diagonal, h, g, f


def test_quadratic_control_matches_hand_completed_square_interval():
    state = fitted()
    query = np.array([[1., 0.], [1., 3.], [-1., 0.]])
    logits = np.column_stack((np.zeros(3), np.array([.2, -.4, .7])))
    actual, diag = predict_ablation(state, query, logits, method='profile_quadratic',
                                   query_ids=('future-0', 'future-1', 'future-2'))
    aa, diagonal, h, g, f = manual_geometry(state, query)
    center = state.map_parameter-np.linalg.solve(h, g)
    qmin = f-.5*(g@np.linalg.solve(h, g))
    expected = []
    for i, coefficient in enumerate(aa.T):
        s = max(0., diagonal[i]-coefficient@coefficient)
        v = coefficient@np.linalg.solve(h, coefficient)+s/state.config.regularization
        radius = np.sqrt(2*(state.threshold-qmin)*v)
        low, high = coefficient@center-radius, coefficient@center+radius
        delta = low if low > 0 else high if high < 0 else 0.
        expected.append(1/(1+np.exp(-(logits[i, 1]+delta))))
    np.testing.assert_allclose(actual[:, 1], expected, atol=1e-13, rtol=1e-12)
    assert diag['not_profile_certified']
    assert not diag['nonlinear_model_set_certificate']
    assert all(row['certificate'] == 'not_profile_certified' for row in diag['queries'])


def test_variance_control_matches_hand_laplace_quadrature_and_r9_like_gate():
    state = fitted()
    query = np.array([[1., 0.], [1., 3.], [-1., 0.]])
    logits = np.column_stack((np.zeros(3), np.array([.2, -.4, .7])))
    actual, diag = predict_ablation(state, query, logits, method='profile_variance_gate')
    aa, diagonal, h, _, _ = manual_geometry(state, query)
    nodes, weights = np.polynomial.hermite.hermgauss(20)
    expected = []
    anchor = 1/(1+np.exp(-logits[:, 1]))
    for i, coefficient in enumerate(aa.T):
        s = max(0., diagonal[i]-coefficient@coefficient)
        variance = s/state.config.regularization+coefficient@np.linalg.solve(h, coefficient)
        prior = diagonal[i]/state.config.regularization
        information = np.clip((prior-variance)/prior, 0., 1.)
        mu = coefficient@state.map_parameter
        latent = logits[i, 1]+mu+np.sqrt(2*variance)*nodes
        pp = (weights@(1/(1+np.exp(-latent))))/np.sqrt(np.pi)
        expected.append((1-.75*information)*anchor[i]+.75*information*pp)
        assert 0 <= variance <= prior+1e-12
    np.testing.assert_allclose(actual[:, 1], expected, atol=1e-13, rtol=1e-12)
    assert diag['queries'][0]['likelihood_scope'] == 'matched_R11_weights_not_native_R9_baseline'
    assert not diag['summary']['true_future_risk_or_coverage_guaranteed']


@pytest.mark.parametrize('method', METHODS)
def test_oneclass_fallback_common_floor_and_no_label_interface(method):
    state = fitted(labels=[0, 0, 0, 0])
    query = np.array([[1., 0.], [-1., 0.]])
    logits = np.array([1000., -1000.])
    output, diag = predict_ablation(state, query, logits, method=method)
    np.testing.assert_array_equal(output, a.anchor_probabilities(logits))
    assert all(row['reason'] == 'one_class_support' for row in diag['queries'])
    assert not any('label' in key for key in inspect.signature(predict_ablation).parameters)
    assert diag['summary']['query_labels_seen'] == 0


@pytest.mark.parametrize('method', METHODS)
def test_zero_cross_returns_anchor_and_fitted_state_is_unchanged(method):
    state = fitted()
    before = {key: getattr(state, key).tobytes() for key in
              ('features', 'factor', 'map_parameter', 'weights', 'margins', 'labels')}
    query = np.array([[0., 2.]])
    logits = np.array([.8])
    output, diag = predict_ablation(state, query, logits, method=method)
    np.testing.assert_array_equal(output, a.anchor_probabilities(logits))
    assert all(getattr(state, key).tobytes() == value for key, value in before.items())
    assert diag['summary']['query_batch_statistics_updated'] is False


@pytest.mark.parametrize('method', METHODS)
def test_query_permutation_other_bad_rows_and_uid_identity(method):
    state = fitted()
    q = np.array([[1., 0.], [-1., 0.]])
    first, _ = predict_ablation(state, q, np.zeros(2), method=method,
                               query_ids=('future-a', 'future-b'))
    second, diag = predict_ablation(state, np.array([[-1., 0.], [np.inf, 2.], [1., 0.]]),
                                   np.zeros(3), method=method,
                                   query_ids=('future-b', 'future-bad', 'future-a'))
    np.testing.assert_array_equal(first, second[[2, 0]])
    np.testing.assert_array_equal(second[1], [.5, .5])
    assert diag['queries'][1]['reason'] == 'ablation_query_geometry_overflow'
    with pytest.raises(ValueError, match='overlap'):
        predict_ablation(state, q[:1], np.zeros(1), method=method,
                         query_ids=('support-0',))


def test_quadratic_c1_is_explicit_anchor_control_not_a_nonlinear_certificate():
    state = fitted(a.Config(amplitude=2., evidence_fraction=1., include_intercept=False))
    output, diag = predict_ablation(state, np.array([[1., 0.]]), np.array([.1]),
                                   method='profile_quadratic')
    np.testing.assert_array_equal(output, a.anchor_probabilities([.1]))
    assert diag['queries'][0]['reason'] == 'explicit_c1_anchor_control'
    assert diag['not_profile_certified'] is True
