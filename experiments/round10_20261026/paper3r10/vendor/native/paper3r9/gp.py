"""Source-anchored finite-support Gaussian-process residual classification.

Only frozen features, support logits and support labels enter fitting. The
Laplace approximation is explicitly approximate; no future-label risk promise
is made. Information anchoring preserves source predictions when the support
kernel conveys zero information about a query. Torch 2.5 / Python 3.12 APIs.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch


@dataclass(frozen=True)
class GPConfig:
    kernel: str = 'linear'
    tau: float = 0.3
    lengthscale: float = 1.0
    jitter: float = 1e-8
    predictive: str = 'anchored_laplace'
    gh_points: int = 20
    anchor_cap: float = 0.75
    max_iterations: int = 60
    tolerance: float = 1e-9

    def validate(self):
        if self.kernel not in ('linear', 'rbf'):
            raise ValueError('Unknown GP kernel')
        if not math.isfinite(self.tau) or self.tau < 0:
            raise ValueError('tau must be finite and nonnegative')
        if not math.isfinite(self.lengthscale) or self.lengthscale <= 0:
            raise ValueError('lengthscale must be finite and positive')
        if not 0 <= self.anchor_cap <= 1:
            raise ValueError('anchor_cap must be in [0,1]')
        if self.predictive not in ('map','laplace','anchored_laplace','constant_laplace','anchored_map'):
            raise ValueError('Unknown predictive mode')
        if not math.isfinite(self.jitter) or not 0 < self.jitter < 1:
            raise ValueError('Relative kernel eigenvalue cutoff must be in (0,1)')
        if self.gh_points < 8 or self.max_iterations < 1:
            raise ValueError('Quadrature and Newton budgets are invalid')
        return self


def _margin(logits):
    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[1] == 2:
        return logits[:,1] - logits[:,0]
    raise ValueError('Expected binary margin [n] or binary logits [n,2]')


def _source_probability(logits):
    return torch.softmax(logits,dim=1) if logits.ndim == 2 else torch.stack((torch.sigmoid(-logits),torch.sigmoid(logits)),1)


@dataclass
class ReferenceGeometry:
    mean: torch.Tensor
    whitener: torch.Tensor
    lengthscale: float
    provenance: dict
    diagnostics: dict

    def transform(self, features):
        return (features - self.mean.to(features)) @ self.whitener.to(features)

    def state_dict(self):
        return {'mean':self.mean.cpu(), 'whitener':self.whitener.cpu(),
                'lengthscale':self.lengthscale, 'provenance':dict(self.provenance),
                'diagnostics':dict(self.diagnostics)}

    @classmethod
    def from_state_dict(cls,state):
        return cls(**state)


def fit_reference_geometry(features, *, weights=None, provenance=None,
                           shrinkage=0.1, max_distance_points=256):
    """Weighted meta_train-only reference, supplied explicitly by the caller.

    Caller owns participant/session weighting and split provenance. Lengthscale
    is a deterministic weighted median of pair distances of at most 256
    evenly indexed reference points. It is never estimated from target queries.
    """
    x=features.detach().to(dtype=torch.float64)
    if x.ndim!=2 or len(x)<2 or not bool(torch.isfinite(x).all()):
        raise ValueError('Reference requires at least two finite feature rows')
    if not 0 < shrinkage <= 1:
        raise ValueError('Reference shrinkage must be in (0,1]')
    if weights is None:
        w=x.new_full((len(x),),1/len(x))
    else:
        w=torch.as_tensor(weights,device=x.device,dtype=x.dtype).detach()
        if w.shape!=(len(x),) or not bool(torch.isfinite(w).all()) or bool((w<0).any()) or not bool(w.sum()>0):
            raise ValueError('Invalid reference weights')
        w=w/w.sum()
    mean=(w[:,None]*x).sum(0)
    centered=x-mean
    covariance=(centered.T*w)@centered
    d=x.shape[1]
    scale=torch.trace(covariance)/d
    floor=max(float(scale)*1e-8,1e-12)
    regularized=(1-shrinkage)*covariance+shrinkage*scale*torch.eye(d,device=x.device,dtype=x.dtype)
    eigen,vectors=torch.linalg.eigh((regularized+regularized.T)/2)
    whitener=(vectors*eigen.clamp_min(floor).rsqrt())@vectors.T
    white=centered@whitener
    m=min(max_distance_points,len(x))
    indices=torch.linspace(0,len(x)-1,m,device=x.device).round().long().unique()
    selected=white[indices]
    distance=torch.cdist(selected,selected).square()/d
    rows,cols=torch.triu_indices(len(indices),len(indices),offset=1,device=x.device)
    distances=distance[rows,cols]
    pairweights=w[indices[rows]]*w[indices[cols]]
    if len(distances) and bool(pairweights.sum()>0):
        order=torch.argsort(distances)
        cumulative=torch.cumsum(pairweights[order].detach().cpu(),0).to(pairweights.device)
        position=torch.searchsorted(cumulative,cumulative[-1]*.5).clamp_max(len(order)-1)
        median=distances[order[position]]
        lengthscale=max(math.sqrt(float(median)),1e-6)
    else:
        lengthscale=1.0
    diag={'reference_rows':len(x),'feature_dimension':d,'shrinkage':shrinkage,
          'effective_weighted_rows':float(1/w.square().sum()),
          'minimum_covariance_eigenvalue':float(eigen.min()),
          'median_distance_points':len(indices),'lengthscale_rule':'weighted_median_pair_distance_sqrt_over_d'}
    return ReferenceGeometry(mean,whitener,lengthscale,dict(provenance or {}),diag)


def _kernel(x,z,config):
    if config.kernel=='linear':
        return config.tau**2*(x@z.T)/x.shape[1]
    distance=torch.cdist(x,z).square()/x.shape[1]
    return config.tau**2*torch.exp(-distance/(2*config.lengthscale**2))


def _kernel_diagonal(x,config):
    if config.kernel=='linear':
        return config.tau**2*x.square().sum(1)/x.shape[1]
    return x.new_full((len(x),),config.tau**2)


def _quadrature_probability(mean,variance,points):
    nodes,weights=np.polynomial.hermite.hermgauss(points)
    nodes=torch.as_tensor(nodes,device=mean.device,dtype=mean.dtype)
    weights=torch.as_tensor(weights,device=mean.device,dtype=mean.dtype)/math.sqrt(math.pi)
    latent=mean[:,None]+torch.sqrt(2*variance.clamp_min(0))[:,None]*nodes
    return (torch.sigmoid(latent)*weights).sum(1)


class GPFit:
    def __init__(self,config,support,projection,mode,precision,diagnostics,reference=None):
        self.config=config
        self.support=support
        self.projection=projection
        self.mode=mode
        self.precision=precision
        self.diagnostics=diagnostics
        self.reference=reference
        self.last_prediction_diagnostics={}

    def state_dict(self):
        return {'config':asdict(self.config),'support':self.support.detach().cpu(),
                'projection':self.projection.detach().cpu(),'mode':self.mode.detach().cpu(),
                'precision':self.precision.detach().cpu(),'diagnostics':dict(self.diagnostics),
                'reference':None if self.reference is None else self.reference.state_dict()}

    @classmethod
    def from_state_dict(cls,state):
        state=dict(state)
        state['config']=GPConfig(**state['config'])
        if state['reference'] is not None:
            state['reference']=ReferenceGeometry.from_state_dict(state['reference'])
        return cls(**state)

    def posterior_moments(self,features):
        x=features.detach().to(dtype=torch.float64)
        if self.reference is not None:
            x=self.reference.transform(x)
        prior=_kernel_diagonal(x,self.config)
        if self.projection.shape[1]==0:
            return torch.zeros_like(prior),prior,torch.zeros_like(prior),prior
        support=self.support.to(x)
        cross=_kernel(x,support,self.config)
        coefficient=cross@self.projection.to(x)
        shift=coefficient@self.mode.to(x)
        solved=torch.linalg.solve(self.precision.to(x),coefficient.T)
        variance=prior-coefficient.square().sum(1)+(coefficient*solved.T).sum(1)
        variance=variance.clamp_min(0)
        # The finite-precision clip implements the mathematical PSD bound.
        variance=torch.minimum(variance,prior)
        information=torch.where(prior>0,(prior-variance)/prior.clamp_min(torch.finfo(x.dtype).tiny),torch.zeros_like(prior)).clamp(0,1)
        return shift,variance,information,prior

    def predict_proba(self,features,base_logits,predictive=None):
        predictive=predictive or self.config.predictive
        if predictive=='posterior':
            predictive='laplace'
        if predictive not in ('map','laplace','anchored_laplace','constant_laplace','anchored_map'):
            raise ValueError('Unknown predictive mode')
        if len(features)!=len(base_logits):
            raise ValueError('Features and logits have different row counts')
        source=_source_probability(base_logits)
        if self.diagnostics.get('fallback_reason'):
            self.last_prediction_diagnostics={'source_fallback':True,'reason':self.diagnostics['fallback_reason']}
            return source
        bad_query=~torch.isfinite(features).all(1)
        clean_features=torch.where(bad_query[:,None],torch.zeros_like(features),features)
        shift,variance,information,prior=self.posterior_moments(clean_features)
        bad_query=bad_query|~(torch.isfinite(shift)&torch.isfinite(variance)&torch.isfinite(information)&torch.isfinite(prior))
        shift=torch.where(bad_query,torch.zeros_like(shift),shift)
        variance=torch.where(bad_query,torch.zeros_like(variance),variance)
        prior=torch.where(bad_query,torch.zeros_like(prior),prior)
        information=torch.where(bad_query,torch.zeros_like(information),information)
        margin=_margin(base_logits).to(dtype=torch.float64)
        if predictive in ('map','anchored_map'):
            probability=torch.sigmoid(margin+shift)
        else:
            probability=_quadrature_probability(margin+shift,variance,self.config.gh_points)
        raw=torch.stack((1-probability,probability),1).to(source)
        gate=(source.new_full((len(source),),self.config.anchor_cap) if predictive=='constant_laplace'
              else self.config.anchor_cap*information.to(source))
        output=(1-gate[:,None])*source+gate[:,None]*raw if predictive in ('anchored_laplace','anchored_map','constant_laplace') else raw
        output=torch.where(bad_query[:,None],source,output)
        self.last_prediction_diagnostics={'source_fallback':False,'predictive':predictive,'nonfinite_query_feature_fallback_rows':int(bad_query.sum()),
            'mean_posterior_variance':float(variance.mean()) if len(variance) else 0,
            'mean_prior_variance':float(prior.mean()) if len(prior) else 0,
            'mean_information_fraction':float(information.mean()) if len(information) else 0,
            'max_information_fraction':float(information.max()) if len(information) else 0,
            'mean_anchor_weight':float(gate.mean()) if len(gate) else 0,
            'max_anchor_weight':float(gate.max()) if len(gate) else 0,
            'max_abs_residual_mean':float(shift.abs().max()) if len(shift) else 0}
        return output

    def predict(self,features,base_logits,mode='posterior'):
        return self.predict_proba(features,base_logits,predictive=mode)


def fit_gp(support_features,base_logits,y,config=None,reference=None):
    """Fit an n-support logistic GP using an exact positive-eigenspace basis.

    Rank is at most n; the objective is ordinary Bernoulli likelihood plus
    0.5||v||² under the Gaussian latent prior. No class weights are silently
    used. Feature/reference preprocessing and all hyperparameters are fixed
    without target query labels. Newton failures return the source prediction
    with an explicit reason; callers retain successful previous sessions.
    """
    config=(config or GPConfig()).validate()
    x=support_features.detach().to(dtype=torch.float64)
    labels=torch.as_tensor(y,device=x.device,dtype=x.dtype).detach()
    margin=_margin(base_logits).detach().to(device=x.device,dtype=x.dtype)
    if x.ndim!=2 or x.shape[1]==0 or len(labels)!=len(x) or len(margin)!=len(x):
        raise ValueError('Support feature, label and logit shapes disagree')
    diag={'support_count':len(x),'feature_dimension':x.shape[1],'kernel':config.kernel,
          'tau':config.tau,'lengthscale':config.lengthscale,'likelihood':'ordinary_Bernoulli',
          'approximation':'Laplace_at_posterior_mode','newton_updates':0,'converged':False,
          'effective_df':0.0,'support_kernel_rank':0,'fallback_reason':None}
    empty=x.new_empty((len(x),0)); zeros=x.new_empty((0,)); identity=x.new_empty((0,0))
    def fallback(reason):
        diag['fallback_reason']=reason
        return GPFit(config,x,empty,zeros,identity,diag,reference)
    if reference is not None:
        x=reference.transform(x)
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(margin).all()) or not bool(torch.isfinite(labels).all()):
        return fallback('nonfinite_support')
    if len(x)==0:
        return fallback('no_valid_support')
    if not bool(((labels==0)|(labels==1)).all()):
        raise ValueError('Binary labels must be zero or one')
    if config.tau==0:
        return fallback('zero_residual_prior')
    if len(torch.unique(labels))<2:
        return fallback('insufficient_class_support')
    matrix=_kernel(x,x,config)
    eigen,vectors=torch.linalg.eigh((matrix+matrix.T)/2)
    threshold=max(float(eigen.max())*config.jitter,torch.finfo(x.dtype).eps)
    diag['kernel_relative_eigenvalue_cutoff']=config.jitter
    keep=eigen>threshold
    rank=int(keep.sum())
    diag['support_kernel_rank']=rank
    if not rank:
        return fallback('zero_support_kernel')
    eigen=eigen[keep]; vectors=vectors[:,keep]
    basis=vectors*torch.sqrt(eigen)
    projection=vectors/torch.sqrt(eigen)
    mode=x.new_zeros(rank)
    eye=torch.eye(rank,dtype=x.dtype,device=x.device)
    def objective(value):
        logits=margin+basis@value
        return (torch.nn.functional.softplus(logits)-labels*logits).sum()+.5*value.square().sum()
    converged=False
    line_search_failed=False
    for iteration in range(config.max_iterations):
        logits=margin+basis@mode
        probability=torch.sigmoid(logits)
        weight=probability*(1-probability)
        gradient=mode+basis.T@(probability-labels)
        precision=eye+(basis.T*weight)@basis
        if float(gradient.abs().max())<=config.tolerance:
            converged=True
            break
        direction=torch.linalg.solve(precision,gradient)
        old=objective(mode)
        descent=gradient@direction
        step=1.0
        accepted=False
        for _ in range(30):
            candidate=mode-step*direction
            value=objective(candidate)
            if bool(torch.isfinite(value)) and bool(value<=old-1e-4*step*descent):
                mode=candidate
                accepted=True
                break
            step*=.5
        diag['newton_updates']=iteration+1
        if not accepted:
            line_search_failed=True
            break
        if float((step*direction).abs().max())<=config.tolerance*(1+float(mode.abs().max())):
            converged=True
            break
    probability=torch.sigmoid(margin+basis@mode)
    weight=probability*(1-probability)
    precision=eye+(basis.T*weight)@basis
    gradient=mode+basis.T@(probability-labels)
    effective_df=rank-torch.trace(torch.linalg.solve(precision,eye))
    diag.update(converged=converged,posterior_mode_norm=float(mode.norm()),
                final_gradient_max=float(gradient.abs().max()),
                effective_df=float(effective_df.clamp(0,rank)),
                negative_log_posterior=float(objective(mode)),
                line_search_failed=line_search_failed)
    if not bool(torch.isfinite(mode).all()) or not bool(torch.isfinite(precision).all()):
        return fallback('nonfinite_newton_state')
    if line_search_failed and float(gradient.abs().max())>1e-5:
        return fallback('newton_line_search_failed')
    # Max-budget finite states are retained and labelled; no training interruption.
    if not converged:
        diag['numerical_note']='finite_newton_state_at_iteration_budget'
    return GPFit(config,x,projection,mode,precision,diag,reference)
