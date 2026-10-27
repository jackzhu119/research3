"""Paired-future score transport (PFST), NumPy-only frozen-feature learning.

Only meta_train episode pairs enter operator training. Target adaptation accepts
support labels alone; prediction is pointwise and never uses query statistics.
The quadratic objective is a global logistic majorizer on the observed training
pairs, not a generalization or future-session guarantee. See R11_MATH.md.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import numpy as np

VERSION = 'paper3-r11-paired-future-score-transport-v1'


@dataclass(frozen=True)
class Config:
    feature_rank: int = 16
    noise_strength: float = 1.
    ridge: float = .01
    operator_mode: str = 'directed'
    noise_type: str = 'permutation'
    training_loss: str = 'majorizer'
    include_intercept: bool = True
    max_support: int = 20
    max_iter: int = 5000
    tolerance: float = 1e-10
    probability_floor: float = 1e-12

    def validate(self):
        if type(self.feature_rank) is not int or not 1 <= self.feature_rank <= 64:
            raise ValueError('feature_rank must be an integer in1..64')
        if self.operator_mode not in ('directed','diagonal','symmetric','psd','identity'):
            raise ValueError('Unknown operator_mode')
        if self.noise_type not in ('permutation','feature','fisher','none'):
            raise ValueError('Unknown support-noise control')
        if type(self.include_intercept) is not bool:raise ValueError('include_intercept must be boolean')
        if self.training_loss not in ('majorizer','erm'):
            raise ValueError('Unknown operator training loss')
        for name in ('noise_strength','ridge','tolerance','probability_floor'):
            value=getattr(self,name)
            if isinstance(value,bool) or not math.isfinite(value):raise ValueError('Finite '+name+' required')
        if self.noise_strength<0 or self.ridge<=0 or self.tolerance<=0:
            raise ValueError('Positive ridge/tolerance and nonnegative noise strength required')
        if not 0<self.probability_floor<1e-3:raise ValueError('Invalid common probability floor')
        if type(self.max_iter) is not int or self.max_iter<1:raise ValueError('Positive max_iter required')
        if type(self.max_support) is not int or not 1<=self.max_support<=20:
            raise ValueError('Maximum paid support budget is20')
        return self


def _sigmoid(value):
    value=np.asarray(value,dtype=np.float64);out=np.empty_like(value);positive=value>=0
    out[positive]=1/(1+np.exp(-value[positive]));exp=np.exp(value[~positive]);out[~positive]=exp/(1+exp)
    return out


def _margins(logits,floor=1e-12):
    logits=np.asarray(logits,dtype=np.float64)
    if not np.isfinite(logits).all():raise ValueError('Finite binary anchor logits required')
    if logits.ndim==2 and logits.shape[1]==2:
        with np.errstate(over='ignore'):logits=logits[:,1]-logits[:,0]
    elif logits.ndim!=1:raise ValueError('Binary logits[n,2] or margins[n] required')
    bound=math.log((1-floor)/floor)
    return np.clip(logits,-bound,bound)


def anchor_probabilities(logits,probability_floor=1e-12):
    if not 0<probability_floor<1e-3:raise ValueError('Invalid probability floor')
    p=np.clip(_sigmoid(_margins(logits,probability_floor)),probability_floor,1-probability_floor)
    return np.column_stack((1-p,p))


def _matrix(value,name,dimension=None):
    value=np.asarray(value,dtype=np.float64)
    if value.ndim!=2 or value.shape[1]<1 or (dimension is not None and value.shape[1]!=dimension):
        raise ValueError(name+' has incompatible frozen feature dimensions')
    if not np.isfinite(value).all():raise ValueError(name+' must be finite')
    return np.ascontiguousarray(value)


def _labels(value,count):
    value=np.asarray(value)
    if value.shape!=(count,) or not np.isin(value,(0,1)).all():raise ValueError('Binary labels matching rows required')
    return value.astype(np.int64)


def _readonly(value):
    value=np.ascontiguousarray(value,dtype=np.float64);value.flags.writeable=False;return value


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


def _identity(metadata,arrays):
    h=hashlib.sha256(_json(metadata).encode())
    for name,a in sorted(arrays.items()):
        a=np.ascontiguousarray(a);h.update(name.encode());h.update(a.dtype.str.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def _reference(reference):
    if not isinstance(reference,dict):raise ValueError('Explicit meta_train feature reference required')
    provenance=reference.get('provenance',{})
    role=reference.get('role',provenance.get('role'))
    if role not in ('meta_train','meta_train_prefix20_only'):
        raise ValueError('Feature reference must be fixed in meta_train')
    mean=np.asarray(reference['mean'],dtype=np.float64)
    if mean.ndim!=1 or not len(mean) or len(mean)>64 or not np.isfinite(mean).all():
        raise ValueError('Finite frozen feature mean of dimension<=64 required')
    w=_matrix(reference['whitener'],'Reference whitener',len(mean))
    if w.shape!=(len(mean),len(mean)):raise ValueError('Square frozen whitening map required')
    axes=_matrix(reference.get('eigenvectors',np.eye(len(mean))),'Reference axes',len(mean))
    if axes.shape!=(len(mean),len(mean)) or not np.allclose(axes.T@axes,np.eye(len(mean)),atol=1e-7,rtol=1e-7):
        raise ValueError('Reference axes must be a fixed orthonormal training basis')
    return {'mean':_readonly(mean),'whitener':_readonly(w),'eigenvectors':_readonly(axes),
            'role':'meta_train','provenance':json.loads(_json(provenance)),
            'coordinate_basis':'meta_train_eigenvectors' if 'eigenvectors' in reference else 'fixed_original_coordinates'}


def feature_map(features,reference,config):
    """Bound feature coordinates first, then append a truly constant offset.

    With intercept, concatenate [bounded_features,1]/sqrt(2); otherwise use
    bounded_features directly. Every row norm is <=1, without batch statistics.
    """
    config.validate();x=_matrix(features,'Frozen features',len(reference['mean']))
    rank=min(config.feature_rank,x.shape[1])
    with np.errstate(over='ignore',invalid='ignore'):
        z=((x-reference['mean'])@reference['whitener'].T)@reference['eigenvectors'][:,:rank]
    if not np.isfinite(z).all():raise ValueError('Frozen feature map overflowed')
    # Stable row norm prevents overflow for large, otherwise finite coordinates.
    scales=np.maximum(1.,np.max(np.abs(z),axis=1)) if len(z) else np.ones(0)
    unit=z/scales[:,None]
    norms=np.sqrt(np.sum(unit*unit,axis=1))
    bounded=unit/np.maximum(1/scales,norms)[:,None]
    if config.include_intercept:return np.column_stack((bounded,np.ones(len(z))))/math.sqrt(2.)
    return bounded


def balanced_weights(labels):
    labels=np.asarray(labels,dtype=np.int64);n=len(labels)
    if not n:return np.empty(0,dtype=np.float64)
    counts=np.bincount(labels,minlength=2)
    if min(counts)==0:return np.full(n,1/n)
    return 1/(2*counts[labels])


def support_moments(phi,logits,labels,config):
    """Exact class-count-conditional label-permutation covariance of the score."""
    phi=_matrix(phi,'Mapped support');n=len(phi);labels=_labels(labels,n)
    z=_margins(logits,config.probability_floor)
    if z.shape!=(n,):raise ValueError('Support logits and feature rows differ')
    p=_sigmoid(z);d=phi.shape[1];counts=np.bincount(labels,minlength=2)
    if n<2 or min(counts)==0:
        return np.zeros(d),np.zeros((d,d)),{'status':'ONE_CLASS_OR_EMPTY_SUPPORT_ANCHOR','n_support':n,'class_counts':counts.tolist()}
    v=balanced_weights(labels);g=phi.T@(v*(labels-p))
    if config.noise_type=='none':noise=np.zeros((d,d))
    elif config.noise_type=='feature':
        # Matched label-agnostic feature covariance: identical to the exact
        # fixed-count permutation covariance on balanced support episodes.
        # Matching centering and scale avoids crediting an implementation
        # mismatch (including an artificial offset penalty) as innovation.
        centered=phi-phi.mean(axis=0)
        noise=(centered.T@centered)/(4*n*(n-1))
    elif config.noise_type=='fisher':noise=(phi.T*(v*v*p*(1-p)))@phi
    else:
        n0,n1=counts
        if n0==n1:
            # For balanced fixed class counts h_i=phi_i/n exactly; the
            # conditional permutation covariance is anchor-independent.
            centered=phi-phi.mean(axis=0)
            noise=(centered.T@centered)/(4*n*(n-1))
        else:
            h=phi*((1-p)/(2*n1)+p/(2*n0))[:,None]
            centered=h-h.mean(axis=0)
            noise=(n1*n0/(n*(n-1)))*(centered.T@centered)
    return g,(noise+noise.T)/2,{'status':'SUPPORT_SCORE_READY','n_support':n,'class_counts':counts.tolist(),
        'score_norm':float(np.linalg.norm(g)),'noise_trace':float(np.trace(noise)),
        'noise_rank':int(np.linalg.matrix_rank(noise,tol=1e-12)),'noise_type':config.noise_type}


def _linear_projection(a,mode):
    if mode=='symmetric':return (a+a.T)/2
    if mode=='diagonal':return np.diag(np.diag(a))
    return a


def _cone_projection(a,mode):
    if mode=='psd':
        vals,vec=np.linalg.eigh((a+a.T)/2);return (vec*np.maximum(vals,0))@vec.T
    if mode=='identity':return np.eye(len(a))*(float(np.trace(a))/len(a))
    return _linear_projection(a,mode)


class _Objective:
    def __init__(self,rows,config):
        self.rows,self.config=rows,config;self.d=rows[0]['g'].shape[0]
        self.b=sum((r['weight']*np.outer(r['r'],r['g']) for r in rows),np.zeros((self.d,self.d)))
        self.lipschitz=config.ridge+.25*sum(r['weight']*np.linalg.norm(r['M'],2)*np.linalg.norm(r['S'],2) for r in rows)

    def hessian(self,a):
        return self.config.ridge*a+.25*sum((r['weight']*(r['M']@a@r['S']) for r in self.rows),np.zeros_like(a))

    def majorizer(self,a):
        return float(.5*np.sum(a*self.hessian(a))-np.sum(self.b*a))

    def erm(self,a,gradient=False):
        value=.5*self.config.ridge*np.sum(a*a);grad=self.config.ridge*a
        for r in self.rows:
            correction=r['phi']@(a@r['g']);z=r['z']+correction
            value+=r['weight']*float(r['v']@(np.logaddexp(0,z)-r['y']*z-r['anchor_loss']))
            grad+=r['weight']*np.outer(r['phi'].T@(r['v']*(_sigmoid(z)-r['y'])),r['g'])
            noise_h=.25*self.config.noise_strength*(r['M']@a@r['N'])
            value+=r['weight']*.5*float(np.sum(a*noise_h));grad+=r['weight']*noise_h
        return (float(value),grad) if gradient else float(value)

    def value_gradient(self,a):
        if self.config.training_loss=='erm':return self.erm(a,True)
        return self.majorizer(a),self.hessian(a)-self.b


def _cg(objective,mode,max_iter,tolerance):
    a=np.zeros_like(objective.b);b=_linear_projection(objective.b,mode);r=b.copy();p=r.copy()
    rr=float(np.sum(r*r));target=tolerance*max(1.,math.sqrt(rr));trace=[];converged=math.sqrt(rr)<=target
    for iteration in range(max_iter):
        if converged:break
        hp=_linear_projection(objective.hessian(p),mode);den=float(np.sum(p*hp))
        if not np.isfinite(den) or den<=0:return np.zeros_like(a),{'solver':'cg','converged':False,'failure':'NONPOSITIVE_OR_NONFINITE_CURVATURE','iterations':iteration}
        step=rr/den;a+=step*p;r-=step*hp;new_rr=float(np.sum(r*r))
        if not np.isfinite(a).all() or not math.isfinite(new_rr):return np.zeros_like(a),{'solver':'cg','converged':False,'failure':'NONFINITE_ITERATE','iterations':iteration+1}
        trace.append(float(objective.majorizer(a)));converged=math.sqrt(new_rr)<=target
        p=r+(new_rr/rr)*p;rr=new_rr
    return _linear_projection(a,mode),{'solver':'matrix_free_cg','converged':converged,
        'iterations':len(trace),'gradient_residual_norm':math.sqrt(rr),'objective_trace':trace}


def _projected(objective,config):
    a=np.zeros_like(objective.b);best=a.copy();best_value=0.;y=a.copy();t=1.;trace=[];converged=False;residual=float(np.linalg.norm(objective.b))
    for iteration in range(config.max_iter):
        _,gradient=objective.value_gradient(y)
        candidate=_cone_projection(y-gradient/objective.lipschitz,config.operator_mode)
        value,_=objective.value_gradient(candidate)
        if not math.isfinite(value) or not np.isfinite(candidate).all():break
        if value<best_value:best_value=value;best=candidate.copy()
        trace.append(value)
        _,g=objective.value_gradient(candidate)
        residual=objective.lipschitz*np.linalg.norm(candidate-_cone_projection(candidate-g/objective.lipschitz,config.operator_mode))
        if residual<=config.tolerance*max(1.,np.linalg.norm(objective.b)):
            converged=True;best=candidate;break
        nt=(1+math.sqrt(1+4*t*t))/2
        y=candidate+(t-1)/nt*(candidate-a);a=candidate;t=nt
    return best,{'solver':'projected_fista','converged':converged,'iterations':len(trace),
                 'objective_trace':trace,'lipschitz_upper_bound':objective.lipschitz,
                 'gradient_mapping_norm':float(residual) if math.isfinite(residual) else None}


@dataclass(frozen=True)
class OperatorState:
    A: np.ndarray
    config: Config
    reference: dict
    diagnostics: dict
    training_identity: str
    operator_identity: str
    status: str='OPERATOR_TRAINED'


@dataclass(frozen=True)
class CalibrationState:
    operator: OperatorState
    g: np.ndarray
    support_ids: tuple
    diagnostics: dict
    fit_identity: str
    status: str='CALIBRATION_READY'

    @property
    def config(self):return self.operator.config

    @property
    def reference(self):return self.operator.reference


def _operator_meta(state):
    return {'version':VERSION,'config':asdict(state.config),'reference_meta':{
        k:v for k,v in state.reference.items() if not isinstance(v,np.ndarray)},
        'diagnostics':state.diagnostics,'training_identity':state.training_identity,'status':state.status}


def _operator_arrays(state):
    return {'A':state.A,'mean':state.reference['mean'],'whitener':state.reference['whitener'],
            'eigenvectors':state.reference['eigenvectors']}


def train_operator(pairs,config=None,reference=None):
    """Learn one operator from actual meta_train support/future pairs only."""
    started=time.perf_counter();config=(config or Config()).validate();reference=_reference(reference)
    pairs=list(pairs)
    if not pairs:raise ValueError('At least one actual meta_train pair required')
    counts={}
    for pair in pairs:
        role=pair.get('training_role',pair.get('role'))
        if role!='meta_train':raise ValueError('Operator training pairs must have training_role meta_train')
        person=str(pair.get('participant_id',''))
        if not person or not str(pair.get('session_id','')):raise ValueError('Training participant/session identity required')
        counts[person]=counts.get(person,0)+1
    rows=[];pair_hashes=[]
    for pair in pairs:
        sx=feature_map(pair['support_features'],reference,config)
        qx=feature_map(pair['future_features'],reference,config)
        if len(sx)>config.max_support:raise ValueError('Support exceeds real first20 acquisition budget')
        sy=_labels(pair.get('support_y',pair.get('support_labels')),len(sx))
        qy=_labels(pair.get('future_y',pair.get('future_labels')),len(qx))
        if not len(qx):raise ValueError('Training pair requires a nonempty frozen future query')
        sz=np.asarray(pair['support_logits'],dtype=np.float64);qz=_margins(pair['future_logits'],config.probability_floor)
        if qz.shape!=(len(qx),):raise ValueError('Future logits/features differ')
        sids=tuple(pair.get('support_ids',()));qids=tuple(pair.get('future_ids',()))
        if (sids and (len(sids)!=len(sx) or len(set(sids))!=len(sids))) or (qids and (len(qids)!=len(qx) or len(set(qids))!=len(qids))) or set(sids)&set(qids):
            raise ValueError('Training support/future identities overlap or duplicate')
        g,n,sd=support_moments(sx,sz,sy,config);v=balanced_weights(qy);p=_sigmoid(qz)
        r=qx.T@(v*(qy-p));m=(qx.T*v)@qx;s=np.outer(g,g)+config.noise_strength*n
        weight=float(pair.get('weight',1/(len(counts)*counts[str(pair['participant_id'])])))
        if not math.isfinite(weight) or weight<=0:raise ValueError('Finite positive training pair weight required')
        row={'g':g,'N':n,'S':s,'r':r,'M':m,'weight':weight,'phi':qx,'z':qz,'v':v,'y':qy,
             'anchor_loss':np.logaddexp(0,qz)-qy*qz,'support_diagnostics':sd}
        rows.append(row)
        pair_hashes.append(_identity({'participant_id':str(pair['participant_id']),'session_id':str(pair['session_id']),
             'support_ids':sids,'future_ids':qids,'weight':weight,'role':'meta_train'},
             {'support_phi':sx,'support_logits':sz,'support_y':sy,'future_phi':qx,'future_margins':qz,'future_y':qy}))
    total=sum(r['weight'] for r in rows)
    for row in rows:row['weight']/=total
    objective=_Objective(rows,config);mode=config.operator_mode
    if config.training_loss=='majorizer' and mode in ('directed','diagonal','symmetric'):
        a,solver=_cg(objective,mode,config.max_iter,config.tolerance)
    elif config.training_loss=='majorizer' and mode=='identity':
        eye=np.eye(objective.d);den=float(np.sum(eye*objective.hessian(eye)));scale=float(np.trace(objective.b))/den
        a=eye*scale;solver={'solver':'unrestricted_scalar_closed_form','converged':True,'iterations':1,'scale':scale}
    else:a,solver=_projected(objective,config)
    value=float(objective.value_gradient(a)[0]);zeroed=False
    # Every finite feasible iterate retains its empirical objective statement;
    # iteration-budget exhaustion does not imply absence of a useful update.
    # Only a broken/nonfinite or worse-than-zero result uses the safe zero.
    if solver.get('failure') or not math.isfinite(value) or value>0.:
        a=np.zeros_like(a);value=0.;zeroed=True
    actual_erm=float(objective.erm(a));majorizer=float(objective.majorizer(a))
    observed_risk=sum(r['weight']*float(r['v']@(np.logaddexp(0,r['z']+r['phi']@(a@r['g']))
                    -r['y']*(r['z']+r['phi']@(a@r['g']))-r['anchor_loss'])) for r in rows)
    training_id=_identity({'pair_hashes':pair_hashes,'training_roles':['meta_train']},{})
    diag={'version':VERSION,'training_pairs':len(rows),'training_participants':len(counts),
          'training_role':'meta_train','training_identity':training_id,
          'parameter_count':int(a.size if mode=='directed' else len(a) if mode=='diagonal'
                                else 1 if mode=='identity' else len(a)*(len(a)+1)//2),
          'stored_operator_entries':int(a.size),
          'mapped_dimension':len(a),'effective_feature_rank':min(config.feature_rank,len(reference['mean'])),
          'objective_value':value,'majorizer_value':majorizer,
          'majorizer_value_includes_ridge':True,
          'ridge_penalty':float(.5*config.ridge*np.sum(a*a)),
          'unregularized_majorizer_value':float(majorizer-.5*config.ridge*np.sum(a*a)),
          'observed_training_erm_difference':actual_erm,'observed_training_balanced_nll_difference':float(observed_risk),
          'erm_difference_includes_ridge_and_noise_penalties':True,
          'zero_operator_objective':0.,'training_objective':config.training_loss,
          'majorizer_nonpositive_certificate':bool(majorizer<=0.),
          'accepted_feasible_iterate':bool(not solver['converged'] and not zeroed),
          'optimality_claimed':bool(solver['converged']),
          'majorizer_dominates_observed_erm':bool(actual_erm<=majorizer+1e-10),
          'fallback_zero_operator':zeroed,'solver':solver,'operator_norm':float(np.linalg.norm(a)),
          'skew_norm':float(np.linalg.norm((a-a.T)/2)),'future_risk_guaranteed':False,
          'certificate_scope':'observed_meta_train_pairs_only','elapsed_seconds':time.perf_counter()-started,
          'coordinate_basis':reference['coordinate_basis'],'support_status_counts':{
              name:sum(r['support_diagnostics']['status']==name for r in rows)
              for name in sorted({r['support_diagnostics']['status'] for r in rows})},
          'balanced_two_class_training_pairs':sum(
              r['support_diagnostics']['class_counts'][0]==r['support_diagnostics']['class_counts'][1]
              and min(r['support_diagnostics']['class_counts'])>0 for r in rows),
          'imbalanced_two_class_training_pairs':sum(
              r['support_diagnostics']['class_counts'][0]!=r['support_diagnostics']['class_counts'][1]
              and min(r['support_diagnostics']['class_counts'])>0 for r in rows)}
    return _make_operator(a,config,reference,diag,training_id,
                          'ZERO_OPERATOR_FALLBACK' if zeroed else 'OPERATOR_TRAINED')


def _make_operator(a,config,reference,diagnostics,training_identity,status='OPERATOR_TRAINED'):
    state=OperatorState(_readonly(a),config,reference,diagnostics,training_identity,'',status)
    # Scientific identity excludes a wall-clock observation; the outer archive
    # SHA still binds every recorded byte, including the timing diagnostic.
    identity_metadata=_operator_meta(state)
    identity_metadata['diagnostics']={key:value for key,value in diagnostics.items() if key!='elapsed_seconds'}
    return OperatorState(state.A,config,reference,diagnostics,training_identity,
                         _identity(identity_metadata,_operator_arrays(state)),status)


def adapt(operator,xS,zS,yS,*,support_ids=None):
    started=time.perf_counter();phi=feature_map(xS,operator.reference,operator.config)
    if len(phi)>operator.config.max_support:raise ValueError('Support exceeds maximum20 acquisition positions')
    ids=tuple(str(x) for x in support_ids) if support_ids is not None else tuple('support:'+str(i) for i in range(len(phi)))
    if len(ids)!=len(phi) or len(set(ids))!=len(ids):raise ValueError('Support identities missing or duplicated')
    g,noise,diag=support_moments(phi,zS,yS,operator.config)
    diag=dict(diag,version=VERSION,operator_identity=operator.operator_identity,
              support_labels_seen=len(phi),query_labels_seen=0,fit_seconds=time.perf_counter()-started)
    fitid=_identity({'operator_identity':operator.operator_identity,'support_ids':ids},
                    {'phi':phi,'logits':np.asarray(zS,dtype=np.float64),'labels':np.asarray(yS,dtype=np.int64),'g':g})
    return CalibrationState(operator,_readonly(g),ids,diag,fitid,diag['status'])


def predict(state,xQ,zQ,*,query_ids=None):
    started=time.perf_counter();phi=feature_map(xQ,state.operator.reference,state.operator.config)
    ids=tuple(str(x) for x in query_ids) if query_ids is not None else ()
    if ids and (len(ids)!=len(phi) or len(set(ids))!=len(ids) or set(ids)&set(state.support_ids)):
        raise ValueError('Query identities duplicate or overlap acquired support')
    cfg=state.operator.config;z=_margins(zQ,cfg.probability_floor)
    if z.shape!=(len(phi),):raise ValueError('Query feature/logit row mismatch')
    delta=phi@(state.operator.A@state.g)
    fallback=~np.isfinite(delta);delta=np.where(fallback,0.,delta)
    p=np.clip(_sigmoid(z+delta),cfg.probability_floor,1-cfg.probability_floor)
    probability=np.column_stack((1-p,p))
    return probability,{'version':VERSION,'operator_identity':state.operator.operator_identity,'fit_identity':state.fit_identity,
        'corrections':delta.tolist(),'summary':{'queries':len(phi),'nonzero_updates':int(np.count_nonzero(delta)),
        'max_abs_correction':float(np.max(np.abs(delta))) if len(delta) else 0.,'numerical_fallback_queries':int(fallback.sum()),
        'query_labels_seen':0,'query_batch_statistics_updated':False,'true_future_risk_or_coverage_guaranteed':False,
        'prediction_seconds':time.perf_counter()-started},'status':state.status}


def _write_npz(path,metadata,arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    complete=dict(metadata);complete['archive_identity']=_identity(metadata,arrays)
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',suffix='.part',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:
            np.savez_compressed(handle,metadata=np.asarray(_json(complete)),**arrays);handle.flush();os.fsync(handle.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)


def _read_npz(path):
    with np.load(path,allow_pickle=False) as f:
        metadata=json.loads(str(f['metadata'].item()));arrays={k:np.asarray(f[k]).copy() for k in f.files if k!='metadata'}
    identity=metadata.pop('archive_identity',None)
    if identity!=_identity(metadata,arrays):raise ValueError('Checkpoint metadata/arrays SHA256 changed')
    return metadata,arrays


def _restore_operator(metadata,arrays):
    if metadata.get('version')!=VERSION:raise ValueError('Operator version differs')
    cfg=Config(**metadata['config']).validate();ref=_reference(dict(metadata['reference_meta'],
             mean=arrays['mean'],whitener=arrays['whitener'],eigenvectors=arrays['eigenvectors']))
    # Preserve the exact recorded coordinate-basis provenance, including safe
    # software fixtures which used original coordinates rather than PCA axes.
    ref['coordinate_basis']=metadata['reference_meta']['coordinate_basis']
    d=min(cfg.feature_rank,len(ref['mean']))+int(cfg.include_intercept)
    if arrays['A'].shape!=(d,d) or not np.isfinite(arrays['A']).all():raise ValueError('Operator shape/value differs')
    state=_make_operator(arrays['A'],cfg,ref,metadata['diagnostics'],metadata['training_identity'],metadata['status'])
    return state


def save_operator(state,path):
    metadata=_operator_meta(state);metadata['operator_identity']=state.operator_identity
    _write_npz(path,metadata,_operator_arrays(state))


def load_operator(path,expected_operator_identity=None):
    metadata,arrays=_read_npz(path);identity=metadata.pop('operator_identity',None)
    state=_restore_operator(metadata,arrays)
    if state.operator_identity!=identity or (expected_operator_identity is not None and identity!=expected_operator_identity):
        raise ValueError('Operator identity mismatch')
    return state


def save_state(state,path):
    meta={'version':VERSION,'kind':'calibration','operator':_operator_meta(state.operator),
          'operator_identity':state.operator.operator_identity,'fit_identity':state.fit_identity,
          'support_ids':state.support_ids,'diagnostics':state.diagnostics,'status':state.status}
    _write_npz(path,meta,dict(_operator_arrays(state.operator),g=state.g))


def load_state(path,expected_fit_identity=None):
    metadata,arrays=_read_npz(path)
    if metadata.get('version')!=VERSION or metadata.get('kind')!='calibration':raise ValueError('Calibration version differs')
    op=_restore_operator(metadata['operator'],arrays)
    if op.operator_identity!=metadata['operator_identity']:raise ValueError('Calibration operator identity differs')
    identity=metadata['fit_identity']
    if expected_fit_identity is not None and identity!=expected_fit_identity:raise ValueError('Calibration fit identity mismatch')
    g=np.asarray(arrays['g'],dtype=np.float64)
    if g.shape!=(len(op.A),) or not np.isfinite(g).all():raise ValueError('Calibration score shape/value differs')
    return CalibrationState(op,_readonly(g),tuple(metadata['support_ids']),metadata['diagnostics'],identity,metadata['status'])
