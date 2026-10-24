"""Finite-window adaptation with explicit head capacity and support risk.

The preceding learners remain available in geometry6. New fits use a common
identity-tangent initializer, then optimize their actual finite-window map.
The support-relative bound is an audit, never a worker-start restriction.
"""
from dataclasses import dataclass, fields, replace
import math
import torch
from . import geometry6

for _name in dir(geometry6):
    if not _name.startswith('__'):
        globals()[_name] = getattr(geometry6, _name)

from . import temporal as _temporal
from . import head_capacity as _head_capacity
from . import risk as _risk

R7_VARIANTS = ('r7_refined',)
VARIANTS = geometry6.VARIANTS + R7_VARIANTS
R7_MODES = _temporal.MODES + ('global_time',)


@dataclass(frozen=True)
class Config(geometry6.Config):
    mode: str = 'spatiotemporal'
    objective: str = 'ce'
    temperature: float = .5
    time_operator: str = 'zero_boundary'
    head_policy: str = 'full'
    head_rank: int = 4
    head_max_rank: int = 16
    head_basis: str = 'pca'
    head_seed: int = 20261024

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError('Unknown round-seven variant')
        preceding = {f.name:getattr(self,f.name) for f in fields(geometry6.Config)}
        preceding['variant'] = self.variant if self.variant in geometry6.VARIANTS else 'geometry_refined'
        geometry6.Config(**preceding)
        if self.mode not in R7_MODES or self.objective not in _risk.KINDS:
            raise ValueError('Unknown finite-window map or support objective')
        if self.time_operator not in _temporal.OPERATORS:
            raise ValueError('Unknown fixed temporal boundary operator')
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('Positive finite support-risk temperature required')
        if self.head_policy not in ('full','budget','fixed','none') or self.head_basis not in ('pca','random'):
            raise ValueError('Unknown support head-capacity policy')
        for value in (self.head_rank,self.head_max_rank):
            if isinstance(value,bool) or not isinstance(value,int) or value < 0:
                raise ValueError('Head ranks must be nonnegative integers')
        if isinstance(self.head_seed,bool) or not isinstance(self.head_seed,int):
            raise ValueError('Head direction seed must be an integer')
        if self.variant in R7_VARIANTS and self.geometry != 'single':
            raise ValueError('Round-seven maps use their explicit mode, not a preceding geometry chart')


@dataclass
class R7Fitted(geometry6.Fitted):
    r7_config: Config | None = None
    head_projector: torch.Tensor | None = None


def _global_theta(theta):
    return torch.cat((theta[:1],theta[1:] * 0))


def transform_signal(geometry,b,weights,x,config):
    """Actual raw signal, with a mathematically exact identity extension.

    The transformed increment lies in CAR. Adding it to x preserves the
    orthogonal complement and avoids an identity whitening/recoloring roundtrip.
    No value-only or detached forward correction is used.
    """
    raw = x.double()
    white = geometry.rref_invsqrt @ geometry.q @ raw
    if config.mode == 'global_time':
        mapped = _temporal.common_time_shift(white,-weights.theta[0]/math.sqrt(10),
                                            config.time_operator)
    else:
        m = (geometry.e @ b @ weights.theta).reshape(10,10)
        mapped = _temporal.apply_white(white,m,config.mode,config.time_operator)
    return raw + geometry.q.T @ geometry.rref_sqrt @ (mapped-white)


def actual_r7(source,geometry,b,weights,x,config):
    signal = transform_signal(geometry,b,weights,x,config)
    features,base = source.features_and_logits(core.joint_trial_rms(signal))
    features,base = features.to(weights.delta_w.dtype),base.to(weights.delta_w.dtype)
    delta = features @ weights.delta_w + weights.delta_b
    return base + torch.stack((-delta/2,delta/2),1),features


def _global_tangent(source,geometry,support,rank,operator):
    """Exact identity derivative for the single active common-time coordinate."""
    core._assert_frozen(source)
    with torch.enable_grad():
        x = support.x.double()
        rms = x.square().mean((-2,-1),keepdim=True).sqrt()
        if len(x) and bool((rms == 0).any()):
            raise ValueError('Zero-energy acquired support trial')
        normalized = (x/rms).detach().requires_grad_(True)
        features,base = source.features_and_logits(normalized)
        margin = base[:,1]-base[:,0]
        gradient = torch.autograd.grad(margin.sum(),normalized,create_graph=False)[0]
        raw_gradient = (gradient-x*(gradient*x).sum((-2,-1),keepdim=True)
                        / x.square().sum((-2,-1),keepdim=True))/rms
        left = geometry.q.T @ geometry.rref_sqrt
        white = geometry.rref_invsqrt @ geometry.q @ x
        d = _temporal.make_time_operator(x.shape[-1],operator,device=x.device,dtype=x.dtype)
        variation = left @ (white @ d.T) / math.sqrt(10)
        j0 = (raw_gradient*variation).sum((-2,-1))
        coordinates = torch.cat((j0[:,None],j0.new_zeros((len(j0),rank-1))),1)
    return margin.detach().double(),coordinates.detach(),features.detach()


def _ridge(weights,config):
    return (config.theta_l2*weights.theta.square().sum()
            + config.head_l2*(weights.delta_w.square().sum()+weights.delta_b.square())
            /(len(weights.delta_w)+1))


def _actual_objective(source,geometry,b,support,weights,config,anchor):
    logits = actual_r7(source,geometry,b,weights,support.x,config)[0]
    return _risk.support_risk(logits,anchor,support.y,kind=config.objective,
                              temperature=config.temperature)+_ridge(weights,config)


def _local_coordinates(weights):
    values = []
    for value in (weights.theta,weights.delta_w,weights.delta_b):
        item = value.clone()
        if not item.requires_grad:
            item = item.detach().requires_grad_(True)
        values.append(item)
    return core.FastWeights(*values)


def _detach_weights(weights):
    return core.FastWeights(*(value.detach() for value in
                              (weights.theta,weights.delta_w,weights.delta_b)))


def _zero_r7(source,b):
    # The new spatial and head optimization share float64 coordinates even
    # when the frozen encoder evaluates in float32.
    return core.FastWeights(b.new_zeros(b.shape[1]),b.new_zeros(source.feature_dim),
                            b.new_zeros(()))


def fit(source,geometry,b,support,config,*,create_graph=False,protected_baseline=None):
    """Fit acquired support only; queries and query labels are not accepted."""
    if config.variant not in R7_VARIANTS:
        return geometry6.fit(source,geometry,b,support,config,create_graph=create_graph,
                             protected_baseline=protected_baseline)
    core._assert_frozen(source)
    if b.shape != (99,config.rank):
        raise ValueError('Dictionary and configured target rank differ')
    if not all(bool((support.y == label).any()) for label in (0,1)):
        return R7Fitted(core.zero_fast_weights(source,b.shape[1],device=b.device),
                        {'mode':config.mode,'optimizer_updates':0},
                        'insufficient_class_support',r7_config=config)
    work_b = b if create_graph else b.detach()
    with torch.enable_grad():
        if config.mode == 'global_time':
            m0,coordinates,features = _global_tangent(source,geometry,support,config.rank,
                                                     config.time_operator)
            fitting_b = torch.eye(config.rank,device=work_b.device,dtype=work_b.dtype)
            j = coordinates
            active_rank = 1
        else:
            m0,coordinates,features = _temporal.support_tangent(source,geometry,support,
                                      mode=config.mode,operator=config.time_operator)
            fitting_b = work_b
            j = coordinates @ work_b
            active_rank = config.rank
        u,head_metadata = _head_capacity.projector(features,config.head_policy,active_rank,
            head_rank=config.head_rank,head_max_rank=config.head_max_rank,
            basis=config.head_basis,seed=config.head_seed)
        reduced_features = features.double() @ u
        p,h = features.shape[1],u.shape[1]
        head_prior = _head_capacity.effective_head_l2(config.head_l2,h,p)
        surrogate_config = replace(config,variant='tangent_profiled',head_l2=head_prior)
        if config.initializer == 'tangent':
            reduced,initial_diagnostics = core._surrogate_fit(m0,coordinates,reduced_features,
                                      fitting_b,support.y,surrogate_config)
            theta = _global_theta(reduced.theta) if config.mode == 'global_time' else reduced.theta
            warm = core.FastWeights(theta,u @ reduced.delta_w,reduced.delta_b)
        else:
            warm = _zero_r7(source,work_b)
            initial_diagnostics = {'surrogate':'zero_fast_weights','optimizer_updates':0}
        zero = _zero_r7(source,work_b)
        anchor = source.source_logits(support.x).detach()
        identity_value = _actual_objective(source,geometry,work_b,support,zero,config,anchor)
        warm_value = _actual_objective(source,geometry,work_b,support,warm,config,anchor)
        if bool(warm_value.detach() <= identity_value.detach()):
            weights,initial_value,chosen = warm,warm_value,config.initializer
        else:
            weights,initial_value,chosen = zero,identity_value,'identity'
        if not create_graph:
            weights = _detach_weights(weights)
        history = [float(initial_value.detach())]
        fractions = []
        for _ in range(config.inner_steps):
            current = _local_coordinates(weights)
            loss = _actual_objective(source,geometry,work_b,support,current,config,anchor)
            gradient = torch.autograd.grad(loss,(current.theta,current.delta_w,current.delta_b),
                                           create_graph=create_graph)
            if not all(bool(torch.isfinite(value).all()) for value in gradient):
                raise FloatingPointError('Nonfinite actual finite-window support gradient')
            next_weights,next_value,accepted = current,loss,0.
            for power in range(config.line_search_steps):
                fraction = 2.**-power
                theta = core.project_ball(current.theta-fraction*config.theta_lr*gradient[0],
                                          config.radius)
                if config.mode == 'global_time':
                    theta = _global_theta(theta)
                head = _head_capacity.project_weights(current.delta_w-fraction*config.head_lr*gradient[1],u)
                candidate = core.FastWeights(theta,head,current.delta_b-fraction*config.head_lr*gradient[2])
                value = _actual_objective(source,geometry,work_b,support,candidate,config,anchor)
                # Exact nonincrease keeps the relative identity anchor at zero;
                # roundoff is reported rather than promoted into a certificate.
                if bool(torch.isfinite(value).detach()) and bool(value.detach() <= loss.detach()):
                    next_weights,next_value,accepted = candidate,value,fraction
                    break
            weights = next_weights if create_graph else _detach_weights(next_weights)
            history.append(float(next_value.detach()))
            fractions.append(accepted)
        if not all(bool(torch.isfinite(value).all()) for value in
                   (weights.theta,weights.delta_w,weights.delta_b)):
            raise FloatingPointError('Nonfinite fitted finite-window parameters')
        if not create_graph:
            weights = _detach_weights(weights)
        actual_logits = actual_r7(source,geometry,work_b,weights,support.x,config)[0]
        diagnostics = dict(initial_diagnostics)
        diagnostics.update(head_metadata)
        diagnostics.update(_risk.risk_diagnostics(actual_logits,anchor,support.y,
                         temperature=config.temperature,ridge=_ridge(weights,config)))
        reduced_head = u.T @ weights.delta_w
        surrogate_margin = m0 + j @ weights.theta + core._augmented(reduced_features) @ torch.cat(
                            (reduced_head,weights.delta_b[None]))
        diagnostics.update(_head_capacity.identifiability(
            j[:,:1] if config.mode == 'global_time' else j,reduced_features,support.y,
            surrogate_margin,config.head_l2,config.theta_l2,full_head_dim=p))
        if config.mode != 'global_time':
            generator = (geometry.e @ work_b @ weights.theta).reshape(10,10)
            diagnostics.update(_temporal.diagnostics(generator,support.x.shape[-1],config.mode,
                                                      config.time_operator))
        else:
            diagnostics.update(global_time_active_parameter_count=1,
                global_time_coefficient=float(weights.theta[0].detach()),
                finite_trial_noncausal=True)
        diagnostics.update(variant=config.variant,mode=config.mode,objective=config.objective,
            temperature=config.temperature,time_operator=config.time_operator,
            initialization_selected_by_actual_support_objective=chosen,
            identity_actual_support_objective=float(identity_value.detach()),
            tangent_actual_support_objective=float(warm_value.detach()),
            actual_support_objective_history=history,actual_line_search_fractions=fractions,
            initial_actual_support_objective=history[0],final_actual_support_objective=history[-1],
            actual_fit_steps=config.inner_steps,
            warm_start_tangent_steps=config.tangent_steps if config.initializer == 'tangent' else 0,
            spatial_parameter_count=active_rank,parameter_container_rank=config.rank,
            theta_norm=float(weights.theta.detach().norm()),
            head_norm=float(weights.delta_w.detach().norm()),
            head_prior_original_dimension=p+1,head_prior_reduced_initializer=head_prior,
            head_subspace_residual=float((weights.delta_w-_head_capacity.project_weights(weights.delta_w,u)).detach().norm()),
            prediction_family='actual_finite_window_map',
            optimizer_scope='actual_support_objective_plus_same_full_head_ridge',
            accepted_branch_derivative='fixed discrete support choices; full continuous first meta derivative',
            query_adaptation_performed=False,identity_forward_extension='raw_signal_plus_CAR_mapped_increment')
    core._assert_frozen(source)
    return R7Fitted(weights,diagnostics,r7_config=config,head_projector=u.detach())


def predict(source,geometry,b,fitted,query_x):
    config = getattr(fitted,'r7_config',None)
    if config is None:
        return geometry6.predict(source,geometry,b,fitted,query_x)
    if fitted.fallback_reason:
        return source.source_logits(query_x)
    return actual_r7(source,geometry,b,fitted.weights,query_x,config)[0]
