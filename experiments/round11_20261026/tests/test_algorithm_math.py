"""Independent mathematical checks: feasible grid, null ambiguity and regret.

The grid enumerates a genuine two-dimensional support parameter region, then
eliminates the query-null completion analytically. It does not reproduce the
production Newton/root algorithm. Tests are CPU-only, synthetic and have no
school/GPU/EEG performance implication.
"""
from dataclasses import replace
import math

import numpy as np

from paper3r11 import r10_profile as a


def _fixture(c=.3, null=True):
    x = np.array([[-1., 0.], [1., 0.]])
    y = np.array([0, 1])
    config = a.Config(amplitude=3., include_intercept=False, evidence_fraction=c,
                      nugget=1e-8, max_newton=80, max_root=12,
                      use_null_completion=null)
    state = a.fit(x, np.zeros((2, 2)), y, config,
                  support_ids=('support-negative', 'support-positive'))
    assert state.status == 'FIT_COMPLETE'
    return state


def _query_geometry(state, q):
    q = np.asarray(q, dtype=float).reshape(1, -1)
    z = a._transform(q, state.reference)
    k = a._kernel(state.features, z, state.config, state.reference)[:, 0]
    coefficient = np.linalg.solve(state.factor, k)
    prior = float(a._kernel_diagonal(z, state.config)[0])
    s = prior-float(coefficient@coefficient)
    return coefficient, max(0., s), prior


def _feasible_grid(state, coefficient, innovation, points=801):
    # Every feasible coordinate lies within this radius because NLL >= 0.
    bound = math.sqrt(2*state.threshold/state.config.regularization)
    coordinate = np.linspace(-bound, bound, points)
    v0, v1 = np.meshgrid(coordinate, coordinate, indexing='ij')
    parameter = np.column_stack((v0.ravel(), v1.ravel()))
    logits = state.margins[None, :] + parameter@state.factor.T
    f = ((np.logaddexp(0., logits)-state.labels[None, :]*logits)@state.weights
         + .5*state.config.regularization*np.sum(parameter*parameter, axis=1))
    feasible = f <= state.threshold
    parameter, f = parameter[feasible], f[feasible]
    assert len(f) > 100
    central = parameter@coefficient
    completion = np.sqrt(np.maximum(0., 2*(state.threshold-f)/state.config.regularization)
                         * innovation)
    return float(np.min(central-completion)), float(np.max(central+completion))


def test_gradient_residual_dual_bounds_enclose_independent_two_dimensional_feasible_grid():
    state = _fixture()
    coefficient, innovation, _ = _query_geometry(state, [1., 0.])
    low, high = _feasible_grid(state, coefficient, innovation)
    upper = a._direction_bound(state, coefficient, innovation)['upper']
    lower = -a._direction_bound(state, -coefficient, innovation)['upper']
    assert lower <= low + 1e-10
    assert upper >= high - 1e-10
    # Dense feasible samples bracket each extremum from the inside. The
    # conservative numerical dual should also be useful, not merely enormous.
    assert low-lower < .02
    assert upper-high < .02
    probability, diagnostics = a.predict(state, np.array([[1., 0.]]),
                                         np.zeros((1, 2)), query_ids=('future-near',))
    p = float(probability[0, 1])
    assert p > .5
    # Regret is affine in eta; checking the independent grid's two attainable
    # endpoint probabilities checks its entire attainable probability range.
    for eta in (1/(1+math.exp(-low)), 1/(1+math.exp(-high))):
        regret = -eta*math.log(p)-(1-eta)*math.log1p(-p)-math.log(2.)
        assert regret <= 1e-9
    assert diagnostics['queries'][0]['certificate'].startswith('analytic_dual')


def test_null_completion_can_veto_update_while_r9_style_information_gate_is_positive():
    state = _fixture()
    q = np.array([[1., 5.]])
    coefficient, innovation, prior = _query_geometry(state, q[0])
    mu = float(coefficient@state.map_parameter)
    witness_cost = state.config.regularization*mu*mu/(2*innovation)
    assert mu > 0 and witness_cost <= state.config.evidence_fraction*state.gain
    probability, diagnostics = a.predict(state, q, np.zeros((1, 2)),
                                         query_ids=('future-unidentified',))
    np.testing.assert_array_equal(probability, np.array([[.5, .5]]))
    assert diagnostics['queries'][0]['reason'] == 'support_null_completion_witness'
    # Calculate the ordinary Laplace information-fraction gate independently.
    # With two equally sized classes, the fit's balanced likelihood is exactly
    # the ordinary likelihood; this is a matched R9-style mathematical
    # counterexample, not a claim that a school R9 task was executed here.
    logits = state.margins+state.factor@state.map_parameter
    p = 1/(1+np.exp(-logits))
    precision = (state.factor.T*(p*(1-p)))@state.factor+np.eye(2)
    variance = innovation+float(coefficient@np.linalg.solve(precision, coefficient))
    information = (prior-variance)/prior
    assert 0 < information < 1
    nodes, weights = np.polynomial.hermite.hermgauss(40)
    latent = mu+np.sqrt(2*variance)*nodes
    pgp = float(np.dot(weights, 1/(1+np.exp(-latent)))/np.sqrt(np.pi))
    r9_gated = .5+.75*information*(pgp-.5)
    assert r9_gated > .5+1e-8
    assert diagnostics['summary']['true_future_risk_or_coverage_guaranteed'] is False


def test_evidence_fraction_nested_sets_move_correction_monotonically_toward_anchor():
    q = np.array([[1., 0.]])
    values = []
    for fraction in (0., .15, .3, .6, .9, 1.):
        state = _fixture(fraction)
        p, _ = a.predict(state, q, np.zeros((1, 2)), query_ids=('future',))
        values.append(float(p[0, 1])-.5)
    assert values[0] > 0
    assert values[-1] == 0
    assert all(left >= right-1e-6 for left, right in zip(values, values[1:]))


def test_existing_frozen_head_can_absorb_all_support_spatial_logits_when_full_row_rank():
    rng = np.random.default_rng(20261026)
    head = rng.normal(size=(15, 65))
    spatial = rng.normal(size=(15, 8))
    projection = head@np.linalg.pinv(head)
    assert np.linalg.matrix_rank(head) == 15
    independent = spatial.T@(np.eye(15)-projection)@spatial
    assert np.linalg.norm(independent) < 1e-10
    arbitrary_target = rng.normal(size=15)
    nuisance = np.linalg.pinv(head)@arbitrary_target
    np.testing.assert_allclose(head@nuisance, arbitrary_target, atol=1e-12, rtol=1e-12)
