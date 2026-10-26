"""Audited strong controls; fitting receives acquired support, never queries.

Historical numerical learners are copied unchanged under vendor/native. The
only R9 source correction is the already deployed CPU metadata cumsum fix.
Missing historical artifacts are explicit errors, never weak substitute fits.
"""
from dataclasses import dataclass, asdict, fields, is_dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import importlib
import json
import sys
import time

import torch

NATIVE = Path(__file__).resolve().parent/'vendor/native'


class MissingBaselineArtifact(ValueError):
    pass


def _native():
    path = str(NATIVE)
    if path not in sys.path:
        sys.path.insert(0, path)
    return {
        'r3': importlib.import_module('paper3r3.algorithm'),
        'r4': importlib.import_module('paper3r4.algorithm'),
        'reference': importlib.import_module('paper3r7.baseline'),
        'head': importlib.import_module('paper3r7.head_capacity'),
        'transport': importlib.import_module('paper3r8.transport'),
        'gp': importlib.import_module('paper3r9.gp'),
        'adapt': importlib.import_module('p3work.adapt'),
        'geometry': importlib.import_module('p3work.spatial_geometry'),
    }


@lru_cache(maxsize=1)
def provenance():
    path=NATIVE.parent/'SOURCE_BASELINES.json'
    manifest=json.loads(path.read_text())
    for row in manifest['files']:
        p=NATIVE.parent/row['vendored_path']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=row['vendored_sha256']:
            raise RuntimeError('Vendored baseline source changed: '+row['vendored_path'])
    return {'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'native_source_versions':{str(x['round'])+':'+Path(x['path']).name:x['ref'] for x in manifest['files']},
            'native_modules_modified_in_place':False}


def _frozen(source):
    check=getattr(source,'assert_unchanged',None)
    if check is not None:
        check()
    elif any(m.training for m in source.modules()) or any(p.requires_grad for p in source.parameters()):
        raise ValueError('Frozen source with eval-mode BN/dropout required')


def _support(support):
    if not hasattr(support,'x') or not hasattr(support,'y'):
        raise ValueError('Support must expose x and y')
    if getattr(support,'role','calibration_prefix') not in ('calibration_prefix','meta_train_support','meta_validation_support'):
        raise ValueError('Only an acquired calibration-support capability may be fitted')
    if support.x.ndim!=3 or support.y.ndim!=1 or len(support.x)!=len(support.y):
        raise ValueError('Expected support [N,C,T] and labels [N]')
    if not bool(torch.isfinite(support.x).all()):
        raise ValueError('Support contains nonfinite EEG')
    if not bool(((support.y==0)|(support.y==1)).all()):
        raise ValueError('Binary acquired support labels required')
    acquired=getattr(support,'acquired_event_uids',None)
    events=getattr(support,'event_uids',None)
    if acquired is not None:
        if len(acquired)>20 or len(set(acquired))!=len(acquired):
            raise ValueError('At most first20 distinct acquired positions may be calibrated')
        if events is not None and (len(events)!=len(support.y) or not set(events)<=set(acquired)):
            raise ValueError('Valid support UIDs must belong to the acquired prefix')
        if events is not None and tuple(u for u in acquired if u in set(events))!=tuple(events):
            raise ValueError('Valid support UIDs must preserve original acquired-prefix order')


def _basis(reference,rank,support):
    b=reference.get('spatial_basis')
    proof=reference.get('basis_provenance')
    if not isinstance(b,torch.Tensor) or b.shape!=(99,rank) or not isinstance(proof,dict) or not proof:
        raise MissingBaselineArtifact('Verified same-seed meta-trained spatial_basis[99,rank] and basis_provenance required')
    if proof.get('query_labels_used') is True or proof.get('future_queries_read') is True or proof.get('holdout_used_for_selection') is True:
        raise ValueError('Spatial baseline dictionary cannot be selected on target queries')
    if proof.get('role',proof.get('training_role')) not in ('meta_train','meta_train_selected_meta_validation','meta_train_only'):
        raise ValueError('Spatial dictionary must be trained on meta_train')
    if not bool(torch.isfinite(b).all()):
        raise ValueError('Nonfinite spatial dictionary')
    return b.detach().to(device=support.x.device,dtype=torch.float64)


def _reference_role(proof,kind):
    if not isinstance(proof,dict) or not proof:
        raise MissingBaselineArtifact('Training-only '+kind+' provenance required')
    if proof.get('future_queries_read') is True or proof.get('query_labels_used') is True or proof.get('holdout_used_for_selection') is True:
        raise ValueError(kind+' cannot use target future queries')
    return proof


def _features(source,geometry,x,ea_state,modules):
    if ea_state is not None and ea_state.get('fallback_reason') is None:
        x=ea_state['transform'].to(x.device)@x.double()
    return source.features_and_logits(modules['geometry'].joint_trial_rms(x))


@dataclass
class BaselineFit:
    method: str
    source: object
    geometry: object
    state: object
    family: str
    diagnostics: dict
    modules: dict
    basis: object=None
    ea_state: object=None
    last_log_probabilities: object=None

    @torch.no_grad()
    def predict_proba(self,query_x):
        """Prediction only: query labels and query-derived fit statistics absent."""
        _frozen(self.source)
        if query_x.ndim!=3:
            raise ValueError('Queries must be an EEG tensor [N,C,T] without labels')
        if self.family=='frozen':
            logits=self.source.source_logits(query_x)
        elif self.family=='reference':
            logits=self.modules['reference'].reference_logits(self.source,self.geometry,self.state,query_x)
        elif self.family=='head':
            if self.diagnostics.get('fallback_reason'):
                logits=self.source.source_logits(query_x)
            else:
                features,base=_features(self.source,self.geometry,query_x,None,self.modules)
                delta=features@self.state['delta_w'].to(features)+self.state['delta_b'].to(features)
                logits=base+torch.stack((-delta/2,delta/2),1)
        elif self.family=='r3':
            logits=self.modules['r3'].predict(self.source,self.geometry,self.basis,self.state,query_x)
        elif self.family=='r4':
            logits=self.modules['r4'].predict(self.source,self.geometry,self.basis,self.state,query_x)
        elif self.family=='transport':
            logits=self.modules['transport'].predict_transport(self.source,self.geometry,self.state,query_x)
        elif self.family=='gp':
            features,logits=_features(self.source,self.geometry,query_x,self.ea_state,self.modules)
            probability=self.state.predict_proba(features,logits)
            self.diagnostics['prediction']=dict(self.state.last_prediction_diagnostics)
            self.last_log_probabilities=probability.double().log()
            if probability.shape!=(len(query_x),2) or not bool(torch.isfinite(probability).all()):
                raise FloatingPointError('Invalid GP baseline probabilities')
            # Native GP output follows the frozen encoder's float32 dtype.
            # Its convex mixture can miss unit row mass by a few float32 ULPs.
            # Preserve those native logs, then represent the same probabilities
            # in float64 with the common row normalization used by the worker.
            row_mass=probability.double().sum(1,keepdim=True)
            error=float((row_mass-1).abs().max()) if len(probability) else 0.
            if (bool((probability<0).any()) or bool((probability>1).any())
                    or error>max(1e-8,8*torch.finfo(probability.dtype).eps)):
                raise FloatingPointError('Invalid GP native probability mass')
            self.diagnostics['prediction'].update(native_probability_dtype=str(probability.dtype),
                native_probability_max_row_mass_error=error,
                probability_representation='native_values_float64_row_normalized')
            probability=probability.double()/row_mass
            _frozen(self.source)
            return probability
        else:
            raise RuntimeError('Unknown fitted baseline family')
        self.last_log_probabilities=torch.log_softmax(logits.double(),1)
        probability=self.last_log_probabilities.exp()
        if probability.shape!=(len(query_x),2) or not bool(torch.isfinite(probability).all()):
            raise FloatingPointError('Invalid baseline probabilities')
        _frozen(self.source)
        return probability

    def state_dict(self):
        """Persist calibration state separately from the frozen source model."""
        if hasattr(self.state,'state_dict'):
            state=self.state.state_dict()
        elif self.family in ('r3','r4','transport'):
            # These dataclasses include tensors and native metadata. They are
            # archived by the worker but not unpickled as arbitrary code.
            state=_archive_state(self.state)
        else:
            state=self.state
        return {'method':self.method,'family':self.family,'state':_archive_state(state),
                'spatial_basis':_archive_state(self.basis),'ea_state':_archive_state(self.ea_state),
                'diagnostics':dict(self.diagnostics),'source_model_included':False}


def _archive_state(value):
    """Convert only known dataclasses/containers; detach tensor autograd leaves."""
    if isinstance(value,torch.Tensor):
        return value.detach().cpu()
    if is_dataclass(value):
        return {f.name:_archive_state(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,dict):
        return {k:_archive_state(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return type(value)(_archive_state(v) for v in value)
    return value


def _head_fit(source,geometry,support,method,config,modules):
    steps=int(config.get('steps',10 if method=='head_sgd10' else 100))
    if steps not in (10,100):
        raise ValueError('Prespecified head-only controls use10 or100steps')
    rank=0 if method=='head_bias' else 4 if method=='head_rank4' else source.feature_dim
    diag={'optimizer_updates':0,'head_steps_requested':steps,'uses_support_labels':True,
          'control':'matched_source_residual_head_SGD','requested_head_rank':rank,
          'head_prior_normalization_dim_including_bias':source.feature_dim+1,
          'head_l2':.001,'head_lr':.01,'fallback_reason':None}
    if not all(bool((support.y==c).any()) for c in(0,1)):
        diag['fallback_reason']='insufficient_class_support'
        return {},diag
    with torch.no_grad():features,base=_features(source,geometry,support.x,None,modules)
    policy='none' if method=='head_bias' else 'fixed' if method=='head_rank4' else 'full'
    u,meta=modules['head'].projector(features,policy,rank=4,head_rank=rank)
    u=u.to(features);z=features@u
    coefficients=torch.zeros(z.shape[1],device=z.device,dtype=z.dtype,requires_grad=True)
    bias=torch.zeros((),device=z.device,dtype=z.dtype,requires_grad=True)
    with torch.enable_grad():
        for _ in range(steps):
            delta=z@coefficients+bias
            logits=base+torch.stack((-delta/2,delta/2),1)
            # Same full physical coefficient ridge after a subspace restriction.
            loss=modules['adapt'].balanced_bce(logits,support.y)+.001*(coefficients.square().sum()+bias.square())/(source.feature_dim+1)
            gw,gb=torch.autograd.grad(loss,(coefficients,bias))
            coefficients=(coefficients-.01*gw).detach().requires_grad_(True)
            bias=(bias-.01*gb).detach().requires_grad_(True)
    diag.update(meta,optimizer_updates=steps,head_parameter_count_including_bias=z.shape[1]+1)
    return {'delta_w':(u@coefficients).detach(),'delta_b':bias.detach(),'head_projector':u.detach()},diag


def fit_baseline(method,source,geometry,support,reference=None,config=None):
    """Fit controls using acquired support only; missing artifacts never weaken it.

    reference keys: spatial_basis+basis_provenance; class_reference+
    class_reference_provenance; r9_reference_raw/ea, r9_config_raw/ea and
    r9_config_provenance_raw/ea. The data layer verifies artifact SHA and seed.
    """
    aliases={'frozen':'source_frozen','ea':'support_EA','support_ea':'support_EA',
             'head_only':'head_sgd10','r9_raw_anchored':'r9_raw_rbf_anchored',
             'r9_ea_anchored':'r9_ea_linear_anchored'}
    method=aliases.get(method,method)
    modules=_native();_frozen(source);_support(support)
    reference=dict(reference or {});config=dict(config or {})
    diag={'baseline':method,'fit_uses_query_inputs':False,'fit_uses_query_labels':False,
          'source_parameters_updated':False,'source_BN_or_dropout_updated':False,
          'implementation':'R10-audited-original-controls-v1','provenance':provenance()}
    started=time.perf_counter();basis=None;ea_state=None
    if method=='source_frozen':
        state=None;family='frozen';native={}
    elif method=='support_EA':
        # Deliberately bypass old task.config.variant; dispatch exact EA API.
        state,native=modules['reference'].fit_reference(source,geometry,support,'support_EA');family='reference'
        assert state['variant']=='support_EA'
    elif method in ('head_bias','head_rank4','head_full','head_sgd10','head_sgd100'):
        state,native=_head_fit(source,geometry,support,method,config,modules);family='head'
    elif method in ('sgd10','sgd100','r3_protected','r3_tangent_protected','r4_xfit_select_stack'):
        base={'rank':4,'radius':.15,'tangent_steps':10,'theta_l2':.01,'head_l2':.1,
              'information_scale':1.,'damping':1e-4,'kl_epsilon':.01}
        base.update(config)
        name='xfit_select_stack' if method=='r4_xfit_select_stack' else 'information_protected' if method=='r3_protected' else 'tangent_protected' if method=='r3_tangent_protected' else method
        if 'variant' in config and config['variant']!=name:
            raise ValueError('Baseline method/config variant mismatch')
        base['variant']=name;basis=_basis(reference,int(base['rank']),support)
        algorithm=modules['r4'] if method=='r4_xfit_select_stack' else modules['r3']
        if method=='r4_xfit_select_stack':
            if not hasattr(support,'acquired_event_uids') or not hasattr(support,'event_uids'):
                raise ValueError('R4 blocked folds require original acquired positions and valid UIDs')
            base={'base_variant':'tangent_profiled','cv_folds':5,'cv_scheme':'blocked','cv_rule':'one_se',
                  'stacking_l2':.01,'cv_head_grid':(.01,.1,1.),'cv_radius_grid':(.05,.15),'gce_q':.7,**base}
        cfg=algorithm.Config(**base)
        state=algorithm.fit(source,geometry,basis,support,cfg,create_graph=False)
        family='r4' if method=='r4_xfit_select_stack' else 'r3';native={**state.diagnostics,'fallback_reason':state.fallback_reason,'native_config':asdict(cfg),'basis_provenance':reference['basis_provenance']}
    elif method=='r8_fiber_bounded':
        contrast=reference.get('class_reference')
        proof=_reference_role(reference.get('class_reference_provenance'),'R8class_reference')
        if not isinstance(contrast,torch.Tensor) or contrast.shape!=(10,10):
            raise MissingBaselineArtifact('R8 requires10x10meta_train-onlyclass_reference')
        role=proof.get('role',proof.get('reference_scope'))
        if role not in ('meta_train','meta_train_first20_acquired_prefix_only','meta_train_prefix20_only'):
            raise ValueError('R8 class reference must come from meta_train prefixes')
        values={'kind':'fiber','alpha':1.,'shrinkage':.05,'covariance_weighting':'pooled',
                'beta':.1,'beta_rule':'bounded','fiber_control':'aligned',**config}
        if values['kind']!='fiber' or values['beta_rule']!='bounded':
            raise ValueError('R8representative must retain fiber/bounded definition')
        cfg=modules['transport'].TransportConfig(**values)
        state=modules['transport'].fit_transport(source,geometry,support,cfg,meta_parameters={'class_reference':contrast})
        family='transport';native={**state.diagnostics,'fallback_reason':state.fallback_reason,'native_config':asdict(cfg),'class_reference_provenance':proof}
    elif method in ('r9_raw_linear_anchored','r9_raw_rbf_anchored','r9_ea_linear_anchored'):
        mode='ea' if method.startswith('r9_ea_') else 'raw'
        expected_kernel='rbf' if method=='r9_raw_rbf_anchored' else 'linear'
        ref=reference.get('r9_reference_'+mode)
        if isinstance(ref,dict):ref=modules['gp'].ReferenceGeometry.from_state_dict(ref)
        if ref is None:
            raise MissingBaselineArtifact('R9 matching'+mode+' meta_trainReferenceGeometry required')
        proof=_reference_role(getattr(ref,'provenance',None),'R9reference')
        if proof.get('role') not in ('meta_train_prefix20_only','meta_train') or proof.get('preprocessing')!=mode:
            raise ValueError('R9 reference role/preprocessing mismatch')
        cfg_values=config or reference.get('r9_config_'+mode)
        config_proof=_reference_role(reference.get('r9_config_provenance_'+mode),'R9config')
        if config_proof.get('selection_role',config_proof.get('role')) not in ('meta_validation','meta_validation_only','meta_validation_fixed_grid'):
            raise ValueError('R9 configuration must be fixed on meta_validation')
        if cfg_values is None:
            raise MissingBaselineArtifact('R9 meta-validation selected GP configuration required')
        if hasattr(cfg_values,'__dataclass_fields__'):cfg_values=asdict(cfg_values)
        values=dict(cfg_values)
        if values.get('kernel',expected_kernel)!=expected_kernel or values.get('predictive','anchored_laplace')!='anchored_laplace':
            raise ValueError('R9 representative requires '+expected_kernel+'/anchored_laplace')
        cfg=modules['gp'].GPConfig(**dict({'kernel':expected_kernel,'predictive':'anchored_laplace'},**values)).validate()
        if mode=='ea':ea_state,_=modules['reference'].fit_reference(source,geometry,support,'support_EA')
        with torch.no_grad():features,base_logits=_features(source,geometry,support.x,ea_state,modules)
        state=modules['gp'].fit_gp(features,base_logits,support.y,config=cfg,reference=ref)
        family='gp';native={**state.diagnostics,'native_config':asdict(cfg),'reference_provenance':proof,'config_provenance':config_proof,'preprocessing':mode}
    else:
        raise ValueError('Unknown prespecified baseline: '+str(method))
    diag.update(native);diag['calibration_seconds']=time.perf_counter()-started
    _frozen(source)
    return BaselineFit(method,source,geometry,state,family,diag,modules,basis,ea_state)


def predict_baseline(method,source,geometry,support,query_x,reference=None,config=None):
    fitted=fit_baseline(method,source,geometry,support,reference=reference,config=config)
    probability=fitted.predict_proba(query_x)
    return probability,dict(fitted.diagnostics)
