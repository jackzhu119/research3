"""Parameter-matched nonlinear rotation/gain coupling for prefix calibration.

The preceding numerical learners remain frozen in ``round4`` and ``core``.
Primary geometry learners refine their own actual nonlinear map on support.
The common identity-tangent fit is retained only as an explicit chart probe.
"""
from dataclasses import dataclass, fields, replace
import math
import torch
from . import round4
from .maps import GEOMETRIES, matrix_map, transform_diagnostics

# The audited controls and their numerical helpers remain available unchanged.
for _name in dir(round4):
    if not _name.startswith('__'):
        globals()[_name] = getattr(round4, _name)

GEOMETRY_VARIANTS = ('geometry_refined', 'geometry_tangent', 'geometry_sgd_zero')
VARIANTS = round4.VARIANTS + GEOMETRY_VARIANTS


@dataclass(frozen=True)
class Config(round4.Config):
    geometry: str = 'single'
    inner_steps: int = 5
    initializer: str = 'tangent'
    max_transform_deviation: float | None = None

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError('Unknown round-five variant')
        previous = {item.name:getattr(self,item.name) for item in fields(round4.Config)}
        previous['variant'] = self.variant if self.variant in round4.VARIANTS else 'tangent_profiled'
        round4.Config(**previous)
        if self.geometry not in GEOMETRIES:
            raise ValueError('Unknown nonlinear geometry')
        if self.inner_steps < 1 or self.initializer not in ('tangent','zero'):
            raise ValueError('Positive inner steps and a registered initializer required')
        if self.geometry == 'bch2' and self.variant in ('geometry_refined','geometry_sgd_zero'):
            raise ValueError('BCH2 is an explicit common-tangent mechanism probe only')
        if self.max_transform_deviation is not None and (
                not math.isfinite(self.max_transform_deviation) or self.max_transform_deviation <= 0):
            raise ValueError('Optional white Frobenius deformation budget must be finite positive')
        if self.variant == 'geometry_tangent' and self.max_transform_deviation is not None:
            raise ValueError('Common-tangent chart probes cannot use a chart-dependent deformation budget')


@dataclass
class Fitted(round4.Fitted):
    # None denotes an unchanged preceding-round learner, not a new map.
    geometry: str | None = None


def geometry_generator(geometry, b, theta):
    return (geometry.e @ b @ theta).reshape(10,10)


def actual_geometry(source, geometry, b, weights, x, chart='single'):
    m = geometry_generator(geometry,b,weights.theta)
    operator = geometry.q.T @ geometry.rref_sqrt @ matrix_map(m,chart) @ geometry.rref_invsqrt @ geometry.q
    features, base = source.features_and_logits(core.joint_trial_rms(operator @ x.double()))
    features, base = features.to(weights.delta_w.dtype), base.to(weights.delta_w.dtype)
    delta = features @ weights.delta_w + weights.delta_b
    return base + torch.stack((-delta/2,delta/2),1), features


def _geometry_loss(source, geometry, b, support, weights, config):
    logits = actual_geometry(source,geometry,b,weights,support.x,config.geometry)[0]
    head_square = weights.delta_w.square().sum()+weights.delta_b.square()
    return (core.balanced_bce(logits,support.y) + config.theta_l2*weights.theta.square().sum()
            + config.head_l2*head_square/(len(weights.delta_w)+1))


def _independent_coordinates(weights):
    # Warm-start Schur heads depend on the solved theta. Clone each coordinate
    # so local support gradients hold the other coordinates fixed, while
    # retaining every warm-start dependency for the first outer derivative.
    values=[]
    for value in (weights.theta,weights.delta_w,weights.delta_b):
        item=value.clone()
        if not item.requires_grad:
            item=item.detach().requires_grad_(True)
        values.append(item)
    return core.FastWeights(*values)


def _feasible(geometry,b,weights,config):
    if config.max_transform_deviation is None:
        return True
    t=matrix_map(geometry_generator(geometry,b,weights.theta),config.geometry)
    eye=torch.eye(len(t),device=t.device,dtype=t.dtype)
    return bool((t-eye).norm().detach() <= config.max_transform_deviation+1e-12)


def fit(source, geometry, b, support, config, *, create_graph=False, protected_baseline=None):
    """Fit acquired support only. No future query tensors or labels accepted."""
    if config.variant not in GEOMETRY_VARIANTS:
        return round4.fit(source,geometry,b,support,config,create_graph=create_graph,
                          protected_baseline=protected_baseline)
    core._assert_frozen(source)
    if b.shape != (99,config.rank):
        raise ValueError('Dictionary and configured rank disagree')
    if not all(bool((support.y==label).any()) for label in (0,1)):
        return Fitted(core.zero_fast_weights(source,b.shape[1],device=b.device),
                      {'geometry':config.geometry,'optimizer_updates':0},
                      'insufficient_class_support', geometry=config.geometry)
    work_b=b if create_graph else b.detach()
    with torch.enable_grad():
        initializer='zero' if config.variant=='geometry_sgd_zero' else config.initializer
        if config.variant=='geometry_tangent':
            initializer='tangent'
        if initializer=='tangent':
            m0,coordinates,features=core.support_tangent(source,geometry,support)
            weights,initial_diagnostics=core._surrogate_fit(m0,coordinates,features,work_b,
                support.y,replace(config,variant='tangent_profiled'))
        else:
            weights=core.zero_fast_weights(source,work_b.shape[1],device=work_b.device)
            initial_diagnostics={'surrogate':'zero_fast_weights'}
        if not _feasible(geometry,work_b,weights,config):
            # The optional matched-amplitude control is mathematical, never a
            # job-start gate. Identity is always feasible; keep the same head.
            for power in range(1,31):
                candidate=core.FastWeights(weights.theta*2.**-power,weights.delta_w,weights.delta_b)
                if _feasible(geometry,work_b,candidate,config):
                    weights=candidate
                    break
            else:
                weights=core.FastWeights(weights.theta*0,weights.delta_w,weights.delta_b)
        history=[float(_geometry_loss(source,geometry,work_b,support,weights,config).detach())]
        fractions=[]
        steps=0 if config.variant=='geometry_tangent' else config.inner_steps
        for _ in range(steps):
            current=_independent_coordinates(weights)
            loss=_geometry_loss(source,geometry,work_b,support,current,config)
            parameters=(current.theta,current.delta_w,current.delta_b)
            gradient=torch.autograd.grad(loss,parameters,create_graph=create_graph)
            if not all(bool(torch.isfinite(value).all()) for value in gradient):
                raise FloatingPointError('Nonfinite actual-geometry support gradient')
            accepted=0.
            next_weights=current
            next_loss=loss
            for power in range(config.line_search_steps):
                fraction=2.**-power
                theta=core.project_ball(current.theta-fraction*config.theta_lr*gradient[0],config.radius)
                candidate=core.FastWeights(theta,current.delta_w-fraction*config.head_lr*gradient[1],
                                          current.delta_b-fraction*config.head_lr*gradient[2])
                if not _feasible(geometry,work_b,candidate,config):
                    continue
                value=_geometry_loss(source,geometry,work_b,support,candidate,config)
                if bool(torch.isfinite(value).detach()) and bool(value.detach() <= loss.detach()+1e-12):
                    next_weights,next_loss,accepted=candidate,value,fraction
                    break
            weights=next_weights
            if not create_graph:
                weights=core.FastWeights(*(value.detach() for value in
                    (weights.theta,weights.delta_w,weights.delta_b)))
            history.append(float(next_loss.detach()))
            fractions.append(accepted)
        if not all(bool(torch.isfinite(value).all()) for value in
                   (weights.theta,weights.delta_w,weights.delta_b)):
            raise FloatingPointError('Nonfinite actual-geometry fitted state')
        if not create_graph:
            weights=core.FastWeights(*(value.detach() for value in
                (weights.theta,weights.delta_w,weights.delta_b)))
        diagnostics=dict(initial_diagnostics)
        diagnostics.update(transform_diagnostics(geometry_generator(geometry,work_b,weights.theta),
            config.geometry,geometry.rref_sqrt,geometry.rref_invsqrt))
        diagnostics.update(variant=config.variant,initializer=initializer,actual_fit_steps=steps,
            warm_start_tangent_steps=config.tangent_steps if initializer=='tangent' else 0,
            total_inner_update_attempts=(config.tangent_steps if initializer=='tangent' else 0)+steps,
            actual_support_loss_history=history,actual_line_search_fractions=fractions,
            initial_actual_support_loss=history[0],final_actual_support_loss=history[-1],
            spatial_parameter_count=config.rank,theta_norm=float(weights.theta.detach().norm()),
            optimizer_scope='actual_chart_balanced_CE_plus_ridge' if steps else 'common_identity_tangent_probe',
            identity_tangent_shared_across_charts=True,
            actual_nonlinear_optimization_performed=bool(steps),
            accepted_branch_derivative='fixed discrete line-search branch; continuous first outer derivative')
    core._assert_frozen(source)
    return Fitted(weights,diagnostics,geometry=config.geometry)


def predict(source, geometry, b, fitted, query_x):
    """Predict through the exact map fitted on support; no query adaptation."""
    chart=getattr(fitted,'geometry',None)
    if chart is None:
        return round4.predict(source,geometry,b,fitted,query_x)
    if fitted.fallback_reason:
        return source.source_logits(query_x)
    return actual_geometry(source,geometry,b,fitted.weights,query_x,chart)[0]
