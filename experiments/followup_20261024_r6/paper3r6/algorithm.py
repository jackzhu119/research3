"""Bounded curvature-shaped prediction averaging for prefix calibration.

This finite projected stencil is not an exact Bayesian posterior. A common
identity-tangent fit supplies its center and a local curvature shape.
"""
from dataclasses import dataclass, fields, replace
from functools import lru_cache
import math
import numpy as np
import torch
from torch.nn import functional as F
from . import geometry5

for _name in dir(geometry5):
    if not _name.startswith('__'):
        globals()[_name] = getattr(geometry5, _name)

UQ_VARIANTS = ('uq_map', 'uq_head', 'uq_linear', 'uq_spatial', 'uq_joint',
               'uq_isotropic', 'uq_independent', 'uq_mean', 'uq_mean_head', 'uq_diagonal')
VARIANTS = geometry5.VARIANTS + UQ_VARIANTS


@dataclass(frozen=True)
class Config(geometry5.Config):
    variance_fraction: float = .25
    head_quadrature: int = 9
    sigma_center_weight: float = .5
    quadrature_rotation: int | None = None

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError('Unknown round-six variant')
        preceding = {item.name:getattr(self,item.name) for item in fields(geometry5.Config)}
        preceding['variant'] = self.variant if self.variant in geometry5.VARIANTS else 'geometry_tangent'
        geometry5.Config(**preceding)
        if not math.isfinite(self.variance_fraction) or self.variance_fraction < 0:
            raise ValueError('Variance fraction must be finite and nonnegative')
        if self.head_quadrature not in (5, 9, 17):
            raise ValueError('Head quadrature must be one of 5, 9 or 17')
        if not math.isfinite(self.sigma_center_weight) or not 0 <= self.sigma_center_weight < 1:
            raise ValueError('Sigma center weight must lie in [0,1)')
        if self.quadrature_rotation is not None and (
                isinstance(self.quadrature_rotation,bool) or not isinstance(self.quadrature_rotation,int)):
            raise ValueError('Quadrature rotation seed must be an integer or None')
        if self.variant in UQ_VARIANTS and self.geometry != 'single':
            raise ValueError('Round-six stencils use the fixed single exponential map')


@dataclass
class UQFitted(geometry5.Fitted):
    uq_config: Config | None = None
    sigma_theta: torch.Tensor | None = None
    sigma_heads: torch.Tensor | None = None
    point_weights: torch.Tensor | None = None
    head_inverse: torch.Tensor | None = None
    spatial_covariance: torch.Tensor | None = None
    head_compensation: torch.Tensor | None = None
    temperature: torch.Tensor | None = None
    create_graph: bool = False


def local_precision(j,z,margin,y,config):
    """CE+ridge precision at the common center, excluding solver damping."""
    _,_,a,cross,d=core._blocks(j,z,margin,y,j.new_zeros(j.shape[1]),
        z.new_zeros(z.shape[1]),replace(config,damping=0.))
    d_inverse=torch.cholesky_inverse(torch.linalg.cholesky((d+d.T)/2))
    compensation=d_inverse @ cross.T
    schur=a-cross @ compensation
    schur=(schur+schur.T)/2
    schur_inverse=torch.cholesky_inverse(torch.linalg.cholesky(schur))
    return a,cross,d,schur,schur_inverse,d_inverse,compensation


def covariance_shape(schur_inverse,radius,fraction,variant):
    """Fix pre-projection trace while controlling spatial covariance shape."""
    temperature=fraction*radius**2/torch.trace(schur_inverse)
    covariance=temperature*schur_inverse
    if variant=='uq_isotropic':
        covariance=torch.eye(len(covariance),device=covariance.device,dtype=covariance.dtype)*torch.trace(covariance)/len(covariance)
    elif variant=='uq_diagonal':
        covariance=torch.diag(covariance.diagonal())
    return (covariance+covariance.T)/2,temperature


def sigma_stencil(theta,covariance,radius,center_weight=.5,rotation=None):
    """Unprojected moments match covariance; projected moments need auditing.

    A finite Cholesky stencil is coordinate-dependent, unlike a Gaussian
    integral. Rotation is a predeclared numerical probe, never query selection.
    """
    rank=len(theta)
    if bool(torch.trace(covariance).detach()==0):
        return theta[None],theta.new_ones(1),theta[None]
    factor=torch.linalg.cholesky(covariance)
    if rotation is not None:
        generator=torch.Generator(device='cpu').manual_seed(rotation)
        matrix=torch.randn(rank,rank,generator=generator,dtype=torch.float64)
        orthogonal,triangular=torch.linalg.qr(matrix)
        orthogonal=orthogonal*torch.where(triangular.diagonal()<0,-1.,1.)
        factor=factor @ orthogonal.to(device=theta.device,dtype=theta.dtype)
    offsets=(math.sqrt(rank/(1-center_weight))*factor).T
    raw=torch.cat((theta[None],theta[None]+offsets,theta[None]-offsets),0)
    points=torch.stack([core.project_ball(point,radius) for point in raw])
    weights=theta.new_full((2*rank+1,),(1-center_weight)/(2*rank))
    weights[0]=center_weight
    return points,weights,raw


@lru_cache(maxsize=3)
def _hermite(order):
    nodes,weights=np.polynomial.hermite.hermgauss(order)
    return tuple(nodes.tolist()),tuple((weights/math.sqrt(math.pi)).tolist())


def normal_logistic_logprob(margin,variance,order=9):
    """Positive Gauss--Hermite rule in log space, without NLL clipping."""
    nodes,weights=_hermite(order)
    nodes,weights=margin.new_tensor(nodes),margin.new_tensor(weights)
    scale=torch.where(variance>0,(2*variance.clamp_min(1e-30)).sqrt(),variance*0)
    values=margin[...,None]+scale[...,None]*nodes
    return torch.stack((torch.logsumexp(F.logsigmoid(-values)+weights.log(),-1),
                        torch.logsumexp(F.logsigmoid(values)+weights.log(),-1)),-1)


def normal_logistic_probability(margin,variance,order=9):
    return normal_logistic_logprob(margin,variance,order)[...,1].exp()


def _symmetric_noise_logits(margin,variance,order):
    """Preserve the exact hard-label null even at floating-point near-ties."""
    logprob=normal_logistic_logprob(margin,variance,order)
    raw=logprob[...,1]-logprob[...,0]
    tiny=torch.finfo(raw.dtype).eps*4
    corrected=torch.where(margin>0,raw.clamp_min(tiny),raw.clamp_max(-tiny))
    corrected=torch.where(margin==0,raw*0,corrected)
    # Correct values only when finite precision loses a mathematical sign.
    # Preserve the quadrature derivative, including its positive slope at zero.
    signed=raw+(corrected-raw).detach()
    return torch.stack((-signed/2,signed/2),-1)


def _head_vector(weights):
    return torch.cat((weights.delta_w.double(),weights.delta_b.double()[None]))


def _actual_components(source,geometry,b,theta,head,x):
    weights=core.FastWeights(theta,head[:-1],head[-1])
    logits,features=geometry5.actual_geometry(source,geometry,b,weights,x,'single')
    return (logits[:,1]-logits[:,0]).double(),core._augmented(features)


def fit(source,geometry,b,support,config,*,create_graph=False,protected_baseline=None):
    """Fit acquired support only; no future query inputs or labels accepted."""
    if config.variant not in UQ_VARIANTS:
        return geometry5.fit(source,geometry,b,support,config,create_graph=create_graph,
            protected_baseline=protected_baseline)
    core._assert_frozen(source)
    if b.shape!=(99,config.rank):
        raise ValueError('Dictionary and rank disagree')
    if not all(bool((support.y==label).any()) for label in (0,1)):
        return UQFitted(core.zero_fast_weights(source,b.shape[1],device=b.device),
            {'optimizer_updates':0,'variant':config.variant},'insufficient_class_support',
            geometry='single',uq_config=config,create_graph=create_graph)
    work_b=b if create_graph else b.detach()
    with torch.enable_grad():
        center_config=replace(config,variant='tangent_profiled')
        m0,coordinates,features=core.support_tangent(source,geometry,support)
        weights,diagnostics=core._surrogate_fit(m0,coordinates,features,work_b,support.y,center_config)
        if not create_graph:
            weights=core.FastWeights(*(item.detach() for item in
                (weights.theta,weights.delta_w,weights.delta_b)))
        diagnostics=dict(diagnostics)
        diagnostics.update(variant=config.variant,geometry='single',
            optimizer_scope='fixed_identity_tangent_common_center',
            prediction_family='bounded_curvature_probability_average',
            exact_bayesian_posterior=False,mean_hessian_used_as_sample_likelihood=False,
            center_solver_damping=config.damping,precision_solver_damping=0.,
            finite_stencil_coordinate_invariant=False,
            spatial_parameter_count=config.rank,theta_norm=float(weights.theta.detach().norm()))
        if config.variant=='uq_map':
            return UQFitted(weights,diagnostics,geometry='single',uq_config=config,create_graph=create_graph)
        j,z=coordinates @ work_b,core._augmented(features)
        head=_head_vector(weights)
        margin=m0+j @ weights.theta+z @ head
        _,_,_,schur,sinv,dinv,compensation=local_precision(j,z,margin,support.y,center_config)
        covariance,temperature=covariance_shape(sinv,config.radius,config.variance_fraction,config.variant)
        points,point_weights,raw=sigma_stencil(weights.theta,covariance,config.radius,
            config.sigma_center_weight,config.quadrature_rotation)
        heads=head[None]-(points-weights.theta[None]) @ compensation.T
        actual_mean=(point_weights[:,None]*points).sum(0)
        head_mean=(point_weights[:,None]*heads).sum(0)
        centered=points-actual_mean
        actual_cov=centered.T @ (point_weights[:,None]*centered)
        diagnostics.update(variance_fraction=config.variance_fraction,
            head_quadrature=config.head_quadrature,sigma_center_weight=config.sigma_center_weight,
            quadrature_rotation=config.quadrature_rotation,
            covariance_trace_before_projection=float(covariance.detach().trace()),
            covariance_trace_after_projection=float(actual_cov.detach().trace()),
            covariance_eigenvalues_before_projection=torch.linalg.eigvalsh(covariance.detach()).tolist(),
            actual_covariance_eigenvalues=torch.linalg.eigvalsh(actual_cov.detach()).tolist(),
            schur_precision_eigenvalues=torch.linalg.eigvalsh(schur.detach()).tolist(),
            generalized_shape_temperature=float(temperature.detach()),
            sigma_points=len(points),projected_point_count=int((raw.detach().norm(dim=1)>config.radius+1e-12).sum()),
            centroid_theta_shift=float((actual_mean-weights.theta).detach().norm()),
            centroid_head_shift=float((head_mean-head).detach().norm()),
            maximum_sigma_theta_norm=float(points.detach().norm(dim=1).max()),
            precision_scope='fixed_identity_tangent_CE_plus_ridge_at_common_center',
            prediction_scope='actual_single_exp_at_projected_stencil',
            independent_pairing_preserves_discrete_marginals=config.variant=='uq_independent')
        values=(points,heads,point_weights,dinv,covariance,compensation,temperature)
        if not create_graph:
            values=tuple(value.detach() for value in values)
        if not all(bool(torch.isfinite(value).all()) for value in values):
            raise FloatingPointError('Nonfinite curvature-shaped stencil state')
    core._assert_frozen(source)
    return UQFitted(weights,diagnostics,geometry='single',uq_config=config,
        sigma_theta=values[0],sigma_heads=values[1],point_weights=values[2],
        head_inverse=values[3],spatial_covariance=values[4],head_compensation=values[5],
        temperature=values[6],create_graph=create_graph)


def _head_variance(z,fitted):
    return fitted.temperature*((z @ fitted.head_inverse)*z).sum(-1).clamp_min(0)


def _linear_prediction(source,geometry,b,fitted,x):
    # Exact local query derivative, holding the head coordinate fixed; no
    # labels, across-query statistics or parameter updates are used.
    with torch.enable_grad():
        theta=fitted.weights.theta.clone()
        if not theta.requires_grad:
            theta=theta.detach().requires_grad_(True)
        head=_head_vector(fitted.weights)
        margin,z=_actual_components(source,geometry,b,theta,head,x)
        # Rank directional JVPs, rather than one reverse pass per future trial.
        # This costs r derivative passes independently of query batch length.
        def margin_at(value):
            return _actual_components(source,geometry,b,value,head,x)[0]
        directions=torch.eye(len(theta),device=theta.device,dtype=theta.dtype)
        jacobian=torch.stack([torch.autograd.functional.jvp(margin_at,theta,direction,
            create_graph=fitted.create_graph)[1] for direction in directions],1)
        residual_gradient=jacobian-z @ fitted.head_compensation
        variance=((residual_gradient @ fitted.spatial_covariance)*residual_gradient).sum(-1)
        variance=variance+_head_variance(z,fitted)
        result=_symmetric_noise_logits(margin,variance.clamp_min(0),fitted.uq_config.head_quadrature)
    return result if fitted.create_graph else result.detach()


def predict(source,geometry,b,fitted,query_x):
    """Finite actual prediction averaging, with no query adaptation."""
    config=getattr(fitted,'uq_config',None)
    if config is None:
        return geometry5.predict(source,geometry,b,fitted,query_x)
    if fitted.fallback_reason:
        return source.source_logits(query_x)
    if config.variant=='uq_map' or config.variance_fraction==0:
        return geometry5.actual_geometry(source,geometry,b,fitted.weights,query_x,'single')[0]
    if config.variant=='uq_linear':
        return _linear_prediction(source,geometry,b,fitted,query_x)
    if config.variant=='uq_head':
        margin,z=_actual_components(source,geometry,b,fitted.weights.theta,_head_vector(fitted.weights),query_x)
        return _symmetric_noise_logits(margin,_head_variance(z,fitted),config.head_quadrature)
    points,heads,weights=fitted.sigma_theta,fitted.sigma_heads,fitted.point_weights
    if config.variant in ('uq_mean','uq_mean_head'):
        theta=(weights[:,None]*points).sum(0)
        head=(weights[:,None]*heads).sum(0)
        margin,z=_actual_components(source,geometry,b,theta,head,query_x)
        if config.variant=='uq_mean':
            return torch.stack((-margin/2,margin/2),1)
        return _symmetric_noise_logits(margin,_head_variance(z,fitted),config.head_quadrature)
    logprobabilities=[]
    log_weights=weights.log()
    for index,theta in enumerate(points):
        if config.variant=='uq_independent':
            # Reuse each spatial EEG forward across every head. This product
            # retains the complete finite marginals, including projection bias.
            margin,z=_actual_components(source,geometry,b,theta,heads[index],query_x)
            source_margin=margin-z @ heads[index]
            all_margins=source_margin[None]+heads @ z.T
            variance=_head_variance(z,fitted)
            conditional=normal_logistic_logprob(all_margins,variance[None],config.head_quadrature)
            logprobabilities.append(torch.logsumexp(log_weights[:,None,None]+conditional,0))
        else:
            margin,z=_actual_components(source,geometry,b,theta,heads[index],query_x)
            if config.variant=='uq_spatial':
                logprobabilities.append(torch.stack((F.logsigmoid(-margin),F.logsigmoid(margin)),-1))
            else:
                logprobabilities.append(normal_logistic_logprob(margin,_head_variance(z,fitted),config.head_quadrature))
    return torch.logsumexp(log_weights[:,None,None]+torch.stack(logprobabilities),0)
