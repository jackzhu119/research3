"""Independent five-source development arms; no internal/external authority.

Plans are read-only. Pure mathematical routines cannot issue production receipts.
Real stages need verified full caches, the actual primary all-method development
freeze, same-seed extension f0, own Rref/banks/meta artifacts, measured engineering
profiles and fixed resource ledgers. No old module global, receipt or stage is
changed. This module contains no networking, credentials or server startup API.
"""
from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass,asdict
from datetime import datetime,timezone
import copy
import fcntl
import hashlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
import weakref
import numpy as np
import torch
from torch import nn
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .contracts import SOURCE_IDS,digest,file_digest,validate_signal_contract
from . import adapt
from .adapt import CalibrationPrefix,PredictionInputs,PredictionResult,FrozenSourceModel,InnerConfig,balanced_bce
from . import spatial_geometry as spatial
from .spatial_geometry import ReferenceCovariance,SourceCovarianceTrial,SpatialGeometry,helmert_rows,spd_roots,tensor_sha256
from .methods import classical,lowrank,neural_controls
from .methods.meta_train import outer_step
from .method_registry import METHOD_IDS,registered_grid,_source_interface
from .meta_training import MetaJob,run_outer_loop,evaluate_development
from .episodes import load_episode_catalog,DeterministicEpisodeSampler,VerifiedEpisodeCatalog
from .selection import choose_checkpoint,participant_equal_metrics,choose_strongest_control
from .source_training import atomic_json,immutable_json,atomic_checkpoint,verified_checkpoint,checkpoint_sidecar
from .training_state import load_checkpoint
from .recovery_budget import PersistentWallBudget
from . import source_extension_bridge as bridge
from .source_extension_bridge import ARMS,INPUT_SHA,BASE_SHA,SEEDS

SCHEMA="paper3-source-extension-downstream-v1"
LEARNED=frozenset({"P","P+PCT","B8","B10","B11"})
DEVELOPMENT_SEEDS=SEEDS[:3]
PRIMARY_METHODS=tuple(METHOD_IDS)
CONTROLS=tuple(f"B{i}" for i in range(1,12))
SAFETY_FACTOR=4.
MAX_ATTEMPTS_PER_TARGET=1
_ISSUER=object()
_CONTEXT_SEALS=weakref.WeakKeyDictionary()
_FIT_SEALS={}
_PREDICTION_SEALS={}
_OWNED_STAGE=False
CODE_FILES=("p3work/source_extension_downstream.py","tools/run_source_extension_downstream.py",
    "p3work/method_registry.py","p3work/contracts.py","p3work/splits.py",
    "p3work/source_extension_bridge.py","p3work/source_extension_training.py",
    "p3work/adapt.py","p3work/spatial_geometry.py","p3work/methods/classical.py",
    "p3work/methods/lowrank.py","p3work/methods/neural_controls.py","p3work/methods/meta_train.py",
    "p3work/meta_training.py","p3work/episodes.py","p3work/selection.py","p3work/training_state.py",
    "p3work/recovery_budget.py","p3work/runtime_clock.py","p3work/training_campaign.py")


class DownstreamError(ValueError):
    """Fixed, non-sensitive production gate failure."""


def _sha(value):
    if not isinstance(value,str) or re.fullmatch(r"[0-9a-f]{64}",value) is None:
        raise DownstreamError("explicit_SHA256_required")
    return value


def _datasets(arm):
    if arm not in ARMS:raise DownstreamError("registered_source_extension_arm_required")
    return frozenset(SOURCE_IDS|{ARMS[arm]})


def _expected_ids(ids):
    ids=frozenset(ids)
    if ids not in {_datasets(a) for a in ARMS}:raise DownstreamError("exact_registered_five_source_arm_required")
    return ids


def _positive(value):return type(value) in (int,float) and math.isfinite(value) and value>0


def local(root,relative,*,exists=True):
    if not isinstance(relative,str) or not relative or Path(relative).is_absolute() or "\\" in relative or ".." in Path(relative).parts:
        raise DownstreamError("safe_root_relative_path_required")
    if any(p.lower().startswith('.env') or p.lower() in {'.aws','.ssh','credentials','secrets','legacy_baseline_v1','external','sealed','internal_holdout'} for p in Path(relative).parts):
        raise DownstreamError("private_legacy_internal_or_external_path_forbidden")
    if Path(relative).suffix.lower() in {'.npz','.pkl','.pickle','.pem','.key'}:
        raise DownstreamError("legacy_or_credential_payload_forbidden")
    root=Path(root).resolve(strict=True);path=root
    for part in Path(relative).parts:
        path/=part
        if path.is_symlink():raise DownstreamError("symlink_forbidden")
    resolved=path.resolve(strict=exists)
    if not resolved.is_relative_to(root):raise DownstreamError("path_escaped_project")
    return resolved


def descriptor(root,path):
    path=Path(path)
    return {'path':path.relative_to(root).as_posix(),'sha256':file_digest(path),'bytes':path.stat().st_size}


def checked(root,item):
    if not isinstance(item,dict):raise DownstreamError('artifact_descriptor_required')
    _sha(item.get('sha256'));path=local(root,item.get('path'))
    if file_digest(path)!=item['sha256'] or 'bytes' in item and path.stat().st_size!=item['bytes']:
        raise DownstreamError('artifact_SHA_or_bytes_changed')
    return path


def _json(path):return json.loads(Path(path).read_text())


def _code(root):
    root=Path(root).resolve(strict=True);executed=Path(__file__).resolve().parents[1]
    result={}
    for name in CODE_FILES:
        path=local(root,name)
        if path.read_bytes()!=(executed/name).read_bytes():raise DownstreamError('executed_code_differs_from_root')
        result[name]=file_digest(path)
    return result


def _uid_sha(uids):return hashlib.sha256('\n'.join(sorted(uids)).encode()).hexdigest()


def build_plan(arm,root=None):
    ids=_datasets(arm)
    meta=[]
    for method in sorted(LEARNED):
        for item in registered_grid(method):
            d=item.get('dimension',item.get('rank'))
            meta.append({'method':method,'configuration':item,'seed':SEEDS[0],'phase':'screening','updates':1000,'dimension':d})
    plan={'schema_version':SCHEMA,'arm':arm,'source_dataset_ids':sorted(ids),
        'methods':list(PRIMARY_METHODS),'seeds':list(DEVELOPMENT_SEEDS),'budget_acquired_trials':20,
        'meta_participants':{'meta_train':40,'meta_validation':11,'internal_holdout_sealed':11},
        'input_contract_sha256':INPUT_SHA,'baseline_manifest_sha256':BASE_SHA,
        'window':{'channels':11,'context_s':[0,1.5],'crop_s':[.25,1.25],'sfreq_hz':128,'n_times':128},
        'meta_screening_jobs':meta,'planned_meta_jobs':55,'screening_validation_jobs':118,
        'additional_confirmation_jobs':28,'source_bank_jobs':6,'meta_engineering_jobs':5,
        'meta_engineering_outer_updates':500,'scientific_meta_outer_updates':55000,
        'scientific_meta_inner_update_upper_bound':2200000,'calibration_profile_settings':118,
        'same_source_for_all_methods':True,'neural_methods_share_same_seed_f0':True,
        'classical_B4_B5_fit_same_source_allowlist':True,'own_reference_and_retrained_meta_parameters':True,
        'primary_all14_development_freeze_required_before_expensive_extension_work':True,
        'budgets':'derived_from_verified_actual_profiles_with_fixed_safety_factor_4_and_shared_arm_cap',
        'target_retry_policy':'committed_fit_reload_only; unknown_dispatched_fit_not_repeated',
        'selection':'participant_BA_desc/balanced_BCE_asc/selected_checkpoint_asc/grid_order',
        'Bstar_rule':'BA_tolerance_0.005_then_same_device_median_calibration_time_then_numeric_method_ID',
        'internal_allowed':False,'external_allowed':False,'primary_H1_changed':False,'actual_fits':0,
        'training_started':False,'server_started':False,'GPU_checked':False}
    if root is not None:plan['code']=_code(root)
    return {**plan,'plan_sha256':digest(plan)}


def source_reference_five(trials,expected_datasets):
    expected=_expected_ids(expected_datasets);q=helmert_rows(11,device='cpu')
    groups={};seen=set()
    for trial in trials:
        if type(trial) is not SourceCovarianceTrial or trial.role!='source_pretrain_train' or trial.dataset not in expected:
            raise PermissionError('Rref_requires_five_source_pretraining_allowlist_only')
        if not trial.subject or not trial.event_uid or trial.event_uid in seen:raise DownstreamError('duplicate_or_missing_source_UID')
        x=trial.x.detach().to(device='cpu',dtype=torch.float64)
        if x.ndim!=2 or x.shape[0]!=11 or not torch.isfinite(x).all():raise DownstreamError('finite_measured_clinic11_required')
        z=q@x;c=z@z.T;t=torch.trace(c)
        if not torch.isfinite(t) or t<=0:raise DownstreamError('zero_source_covariance')
        groups.setdefault(trial.dataset,{}).setdefault(str(trial.subject),[]).append(c/t);seen.add(trial.event_uid)
    if set(groups)!=set(expected):raise DownstreamError('all_five_source_datasets_required')
    mean=torch.stack([torch.stack([torch.stack(values).mean(0) for _,values in sorted(groups[d].items())]).mean(0) for d in sorted(groups)]).mean(0)
    matrix=.95*mean+.05*torch.eye(10,dtype=torch.float64)/10
    root,inverse=spd_roots(matrix);uids=tuple(sorted(seen))
    return ReferenceCovariance(matrix,root,inverse,uids,_uid_sha(uids))


@dataclass(frozen=True)
class MathContext:
    source: FrozenSourceModel
    reference: ReferenceCovariance
    arm: str
    seed: int
    signal_sha: str
    split_sha: str
    def __post_init__(self):
        _datasets(self.arm);_sha(self.signal_sha);_sha(self.split_sha)
        if type(self.source) is not FrozenSourceModel or self.seed not in SEEDS:raise DownstreamError('same_seed_frozen_source_required')
        if type(self.reference) is not ReferenceCovariance or self.reference.matrix.shape!=(10,10):raise DownstreamError('own_clinic11_reference_required')
        self.source.assert_unchanged();self.geometry.validate()
    @property
    def geometry(self):
        device=next(self.source.parameters()).device
        base=SpatialGeometry.from_reference(self.reference.matrix.detach().to('cpu'))
        return SpatialGeometry(*(value.to(device) for value in (base.q,base.rref_sqrt,base.rref_invsqrt,base.e)))
    @property
    def identity(self):
        self.source.assert_unchanged()
        return digest({'arm':self.arm,'seed':self.seed,'source':self.source.frozen_state_sha256,
            'source_interface':_source_interface(self.source),
            'geometry':self.geometry.sha256(),'signal_sha':self.signal_sha,'split_sha':self.split_sha})


@dataclass(frozen=True)
class BankPayload:
    method: str
    arrays: dict
    metadata: dict
    state_sha256: str
    def assert_unchanged(self):
        if _bank_sha(self.method,self.arrays,self.metadata)!=self.state_sha256:raise DownstreamError('source_bank_payload_changed')


def _bank_sha(method,arrays,metadata):
    return digest({'method':method,'metadata':metadata,'arrays':{k:classical.array_sha256(np.asarray(v)) for k,v in sorted(arrays.items())}})


def _bank(method,arrays,metadata):
    arrays={k:np.asarray(v,dtype=np.float64).copy() for k,v in arrays.items()}
    for value in arrays.values():value.flags.writeable=False
    return BankPayload(method,arrays,copy.deepcopy(metadata),_bank_sha(method,arrays,metadata))


def fit_source_bank_math(x,y,rows,method,configuration,expected_datasets,signal_sha):
    expected=_expected_ids(expected_datasets);_sha(signal_sha)
    if method not in {'B4','B5'}:raise DownstreamError('B4_or_B5_source_bank_required')
    if any(r.get('role') not in {'source_train','source_pretrain_train'} or r.get('dataset') not in expected for r in rows):
        raise PermissionError('source_bank_accepts_source_only_roles')
    if {r['dataset'] for r in rows}!=set(expected):raise DownstreamError('bank_all_five_sources_required')
    x=np.asarray(x);y=np.asarray(y)
    if x.ndim!=3 or x.shape[1:]!=(11,128) or len(rows)!=len(x) or y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!={0,1}:
        raise DownstreamError('source_bank_shape_and_LR_labels_required')
    uids=[r.get('event_uid') for r in rows]
    if any(not isinstance(u,str) or not u for u in uids) or len(set(uids))!=len(uids) or any(r.get('label',int(v))!=int(v) for r,v in zip(rows,y)):
        raise DownstreamError('source_bank_UID_or_label_identity_invalid')
    if not np.isfinite(x).all() or np.any(np.sum(x*x,axis=(1,2))<=0):raise DownstreamError('bank_nonfinite_or_zero_source')
    q=helmert_rows().numpy();meta={'scope':'MATHEMATICAL_PAYLOAD_NOT_SCIENTIFIC_RECEIPT','source_dataset_ids':sorted(expected),
        'signal_sha':signal_sha,'UID_set_sha256':_uid_sha(uids),'n_times':128,'source_trials':len(x),'converged':True}
    if method=='B4':
        m=configuration.get('m')
        if m not in (1,2,3) or set(configuration)!={'m'}:raise DownstreamError('source_CSP_m_required')
        means=classical.hierarchical_class_means(classical.trace_covariances(x),rows,y)
        filters,eigenvalues=classical.csp_filters(means,m);features=classical.csp_features(x,filters)
        clf=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=[.5,.5]).fit(features,y)
        meta.update(m=m,exact_solves=1,optimizer_updates=0)
        return _bank(method,{'q':q,'class_means':means,'filters':filters,'eigenvalues':eigenvalues,'coef':clf.coef_.reshape(-1),'bias':clf.intercept_[0]},meta)
    shrink=configuration.get('shrinkage',configuration.get('shrink'))
    if shrink not in (.01,.05,.1) or len(configuration)!=1:raise DownstreamError('source_TS_shrinkage_required')
    weights=classical.hierarchical_weights(rows,y);cov=classical.trace_covariances(x,shrink=shrink)
    ref=classical.matrix_function(np.average(classical.matrix_function(cov,'log'),axis=0,weights=weights),'exp')
    inverse=classical.matrix_function(ref,'inverse_root');features=classical.tangent_features(cov,inverse)
    scaler=StandardScaler().fit(features,sample_weight=weights)
    clf=LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,random_state=0)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always',ConvergenceWarning);clf.fit(scaler.transform(features),y,sample_weight=weights*len(y))
    meta.update(shrink=shrink,converged=not any(issubclass(w.category,ConvergenceWarning) for w in observed),optimizer_updates=int(clf.n_iter_[0]))
    return _bank(method,{'q':q,'reference':ref,'inverse_root':inverse,'scaler_mean':scaler.mean_,'scaler_scale':scaler.scale_,'coef':clf.coef_.reshape(-1),'bias':clf.intercept_[0]},meta)


@dataclass(frozen=True)
class AdapterState:
    method: str
    configuration: dict
    context_sha256: str
    payload: dict
    dictionary: torch.Tensor|None
    updates: int
    fallback_reason: str|None
    trainable_parameters: int
    support_event_uids: tuple
    acquired_event_uids: tuple
    state_sha256: str


def _tensor_tree_sha(values):
    output={}
    for key,value in sorted(values.items()):
        if isinstance(value,torch.Tensor):
            value=value.detach().cpu().contiguous()
            output[key]={'shape':list(value.shape),'dtype':str(value.dtype),
                'sha':hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
        elif isinstance(value,np.ndarray):output[key]={'np':classical.array_sha256(value)}
        elif isinstance(value,dict):output[key]=_tensor_tree_sha(value)
        else:output[key]=value
    return output


def _adapter_sha(method,row,ctx,payload,dictionary,updates,fallback,count,support,acquired):
    return digest({'method':method,'configuration':row,'context':ctx,'payload':_tensor_tree_sha(payload),
        'dictionary':tensor_sha256(dictionary) if dictionary is not None else None,'updates':updates,
        'fallback':fallback,'parameters':count,'support':support,'acquired':acquired})


def _adapter(ctx,support,method,row,payload,*,dictionary=None,updates=0,fallback=None,count=0):
    context=ctx.identity
    return AdapterState(method,copy.deepcopy(row),context,payload,dictionary,updates,fallback,count,
        support.event_uids,support.acquired_event_uids,_adapter_sha(method,row,context,payload,dictionary,updates,fallback,count,support.event_uids,support.acquired_event_uids))


def _bank_target(bank,support,row):
    bank.assert_unchanged();x=support.x.detach().cpu().numpy();y=support.y.cpu().numpy()
    arrays={k:v.copy() for k,v in bank.arrays.items()};meta=copy.deepcopy(bank.metadata)
    if set(y.tolist())!={0,1}:return {'arrays':arrays,'metadata':meta},0,'insufficient_class_support',0
    if bank.method=='B4':
        if row['m']!=meta['m']:raise DownstreamError('B4_target_m_differs_from_own_bank')
        cov=classical.trace_covariances(x);means=np.stack([cov[y==c].mean(0) for c in (0,1)])
        mixed=(1-row['alpha'])*means+row['alpha']*arrays['class_means']
        filters,eigen=classical.csp_filters(mixed,row['m']);features=classical.csp_features(x,filters)
        try:clf=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=[.5,.5]).fit(features,y)
        except (ValueError,np.linalg.LinAlgError):return {'arrays':arrays,'metadata':meta},0,'lda_numerical_fit_failed',0
        arrays.update(filters=filters,eigenvalues=eigen,coef=clf.coef_.reshape(-1),bias=clf.intercept_[0])
        meta.update(exact_solves=1,converged=True)
        return {'arrays':arrays,'metadata':meta},0,None,int((filters.size if row['alpha']<1 else 0)+2*row['m']+1)
    if row['shrinkage']!=meta['shrink']:raise DownstreamError('B5_target_shrinkage_differs_from_own_bank')
    cov=classical.trace_covariances(x,shrink=meta['shrink'])
    features=(classical.tangent_features(cov,arrays['inverse_root'])-arrays['scaler_mean'])/arrays['scaler_scale']
    start=np.concatenate([arrays['coef'],[float(arrays['bias'])]])
    result=minimize(classical.balanced_logistic_objective,start.copy(),args=(features,y,start,row['l2_to_source']),jac=True,method='L-BFGS-B',options={'maxiter':200,'gtol':1e-8,'ftol':0.,'maxls':50})
    arrays.update(coef=result.x[:-1],bias=result.x[-1]);meta.update(converged=bool(result.success and np.max(np.abs(result.jac))<=1e-8),function_evaluations=int(result.nfev))
    return {'arrays':arrays,'metadata':meta},int(result.nit),None,56


def fit_support_math(ctx,support,method,configuration,learned=None,bank=None):
    """Pure support math; this result has no scientific completion authority."""
    if torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda'):
        raise DownstreamError('mixed_precision_forbidden')
    if not isinstance(ctx,MathContext) or type(support) is not CalibrationPrefix:raise TypeError('typed_context_and_CalibrationPrefix_required')
    support.__post_init__();ctx.source.assert_unchanged()
    if support.role!='meta_validation_support':raise PermissionError('extension_development_support_only')
    if support.protocol_freeze_sha!=ctx.signal_sha or support.x.shape[1:]!=(11,128):raise DownstreamError('support_signal_contract_mismatch')
    row=next((r for r in registered_grid(method) if r==configuration),None)
    if row is None:raise DownstreamError('registered_method_configuration_required')
    if method in LEARNED and learned is None:raise DownstreamError('own_independent_learned_parameter_required')
    if method not in LEARNED and learned is not None:raise DownstreamError('fixed_control_cannot_accept_learned_parameter')
    if method not in {'B4','B5'} and bank is not None:raise DownstreamError('neural_method_cannot_accept_classical_bank')
    device=next(ctx.source.parameters()).device
    geometry=ctx.geometry
    if method in {'B4','B5'}:
        if type(bank) is not BankPayload or bank.method!=method or set(bank.metadata['source_dataset_ids'])!=set(_datasets(ctx.arm)) or bank.metadata['signal_sha']!=ctx.signal_sha:
            raise DownstreamError('same_arm_five_source_classical_bank_required')
        payload,updates,fallback,count=_bank_target(bank,support,row)
        return _adapter(ctx,support,method,row,payload,updates=updates,fallback=fallback,count=count)
    if method=='B2':
        config=neural_controls.FineTuneConfig(**row)
        fit=neural_controls.fit_neural_control_development(ctx.source,support,method='B2',config=config)
        payload={'model':{k:v.detach().clone() for k,v in fit.target_model.state_dict().items()}} if fit.target_model is not None else {}
        return _adapter(ctx,support,method,row,payload,updates=fit.updates,fallback=fit.fallback_reason,count=fit.trainable_parameters)
    both=bool((support.y==0).any() and (support.y==1).any())
    if method=='B3':
        count=ctx.source.feature_dim+1
        if not both:return _adapter(ctx,support,method,row,{},fallback='insufficient_class_support',count=count)
        alignment,cov,shrunk=neural_controls._support_alignment(geometry,support,shrinkage=row['shrinkage'],role='meta_validation_support')
        with torch.no_grad():features,base=ctx.source.features_and_logits(spatial.joint_trial_rms(alignment@support.x.detach().to(torch.float64)))
        dw=features.new_zeros(ctx.source.feature_dim,requires_grad=True);db=features.new_zeros((),requires_grad=True)
        for _ in range(10):
            margin=features@dw+db;logits=base+torch.stack([-margin/2,margin/2],1)
            loss=balanced_bce(logits,support.y)+.001*(dw.square().sum()+db.square())/count
            gw,gb=torch.autograd.grad(loss,(dw,db));dw=(dw-row['head_learning_rate']*gw).detach().requires_grad_(True);db=(db-row['head_learning_rate']*gb).detach().requires_grad_(True)
        return _adapter(ctx,support,method,row,{'alignment':alignment.detach(),'delta_w':dw.detach(),'delta_b':db.detach(),'target_covariance':cov,'target_shrunk_covariance':shrunk},updates=10,count=count)
    if method in {'B6','B7','B10','B11'}:
        u=learned.detach().clone().to(device=device,dtype=torch.float64) if learned is not None else lowrank.initialize_lowrank(row['rank'],ctx.seed,device=device)
        config=lowrank.LowRankConfig(**row,pct_weight=.1 if method in {'B7','B11'} else 0.)
        weights,updates,fallback=lowrank.inner_unroll_lowrank(ctx.source,geometry,u,support,config=config)
        payload={k:getattr(weights,k).detach().clone() for k in ('u','v','delta_w','delta_b')}
        return _adapter(ctx,support,method,row,payload,dictionary=u,updates=updates,fallback=fallback,count=20*row['rank']+ctx.source.feature_dim+1)
    rank=row.get('dimension',2)
    dictionary=learned.detach().clone().to(device=device,dtype=torch.float64) if learned is not None else spatial.initialize_dictionary(rank,ctx.seed,device=device)
    geometry.validate_dictionary(dictionary)
    if dictionary.shape[1]!=rank:raise DownstreamError('learned_dictionary_configuration_mismatch')
    config=InnerConfig(radius=row.get('radius',.15),head_lr=row.get('head_lr',.01),head_l2=row.get('head_l2',.001),pct_weight=.1 if method=='P+PCT' else 0.)
    if method=='B0':weights=adapt.zero_fast_weights(ctx.source,rank,device=device);updates=0;fallback=None
    else:weights,updates,fallback=adapt.inner_unroll(ctx.source,geometry,dictionary,support,config=config,family='additive' if method=='B8' else 'exp',head_only=method=='B1')
    payload={k:getattr(weights,k).detach().clone() for k in ('theta','delta_w','delta_b')}
    count=0 if method=='B0' else ctx.source.feature_dim+1+(0 if method=='B1' else rank)
    return _adapter(ctx,support,method,row,payload,dictionary=dictionary,updates=updates,fallback=fallback,count=count)


def predict_math(ctx,state,inputs):
    if torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda'):
        raise DownstreamError('mixed_precision_forbidden')
    if not isinstance(ctx,MathContext) or type(state) is not AdapterState or type(inputs) is not PredictionInputs:raise TypeError('typed_unlabeled_prediction_required')
    inputs.__post_init__();ctx.source.assert_unchanged()
    if inputs.x.shape[1:]!=(11,128) or inputs.protocol_freeze_sha!=ctx.signal_sha or state.context_sha256!=ctx.identity:
        raise DownstreamError('prediction_context_input_or_arm_changed')
    if set(inputs.event_uids)&set(state.acquired_event_uids):raise DownstreamError('query_overlaps_acquired_prefix_including_failed_QC')
    expected=_adapter_sha(state.method,state.configuration,state.context_sha256,state.payload,state.dictionary,state.updates,state.fallback_reason,state.trainable_parameters,state.support_event_uids,state.acquired_event_uids)
    if state.state_sha256!=expected:raise DownstreamError('fitted_parameter_payload_changed')
    method=state.method;p=state.payload
    with torch.no_grad():
        if method in {'B4','B5'}:
            arrays=p['arrays'];x=inputs.x.detach().cpu().numpy()
            if method=='B4':features=classical.csp_features(x,arrays['filters'])
            else:features=(classical.tangent_features(classical.trace_covariances(x,shrink=p['metadata']['shrink']),arrays['inverse_root'])-arrays['scaler_mean'])/arrays['scaler_scale']
            margin=features@arrays['coef']+arrays['bias'];logits=torch.from_numpy(np.stack([np.zeros_like(margin),margin],1)).to(inputs.x.device)
        elif state.fallback_reason or method=='B0':logits=ctx.source.source_logits(inputs.x)
        elif method=='B2':
            target=neural_controls._EvalSourceCopy(ctx.source);target.load_state_dict(p['model'],strict=True)
            for parameter in target.parameters():parameter.requires_grad_(False)
            logits=target(inputs.x)
        elif method=='B3':
            features,base=ctx.source.features_and_logits(spatial.joint_trial_rms(p['alignment']@inputs.x.to(torch.float64)))
            margin=features@p['delta_w']+p['delta_b'];logits=base+torch.stack([-margin/2,margin/2],1)
        elif method in {'B6','B7','B10','B11'}:
            logits=lowrank.lowrank_logits(ctx.source,ctx.geometry,inputs.x,lowrank.LowRankWeights(*(p[k] for k in ('u','v','delta_w','delta_b'))))
        else:
            weights=adapt.FastWeights(*(p[k] for k in ('theta','delta_w','delta_b')))
            logits=adapt.adapted_logits(ctx.source,ctx.geometry,state.dictionary,inputs.x,weights,family='additive' if method=='B8' else 'exp')
        probability=torch.softmax(logits,dim=1);predicted=logits.argmax(1)
    ctx.source.assert_unchanged()
    return PredictionResult(logits,probability,predicted,inputs.event_uids,inputs.protocol_freeze_sha)


def _catalog(root,value):
    root=Path(root).resolve(strict=True)
    if isinstance(value,dict) and {'path','sha256'}<=set(value):path=checked(root,value);value=_json(path)
    elif isinstance(value,(str,Path)):path=local(root,str(value));value=_json(path)
    if not isinstance(value,dict) or value.get('schema_version')!=SCHEMA or value.get('arm') not in ARMS:
        raise DownstreamError('independent_extension_artifact_catalog_required')
    if value.get('input_contract_sha256')!=INPUT_SHA or value.get('internal_allowed') is not False or value.get('external_allowed') is not False:
        raise DownstreamError('catalog_signal_or_permission_scope_invalid')
    return copy.deepcopy(value)


def _primary(root,catalog):
    from .internal_validation import verify_freeze
    ref=catalog.get('primary_development_freeze')
    checked(root,ref)
    frozen=verify_freeze(root,ref)
    if (frozen.get('status')!='IMMUTABLE_PRE_INTERNAL_DEVELOPMENT_FREEZE'
            or set(frozen.get('method_order',[]))!=set(PRIMARY_METHODS)
            or frozen.get('seeds')!=list(DEVELOPMENT_SEEDS)
            or frozen.get('internal_data_used_for_selection') is not False):
        raise DownstreamError('actual_primary_all14_three_seed_Bstar_freeze_required')
    return frozen


def _source_inputs(root,catalog,seed,*,primary_required=True):
    root=Path(root).resolve(strict=True);catalog=_catalog(root,catalog)
    if primary_required:_primary(root,catalog)
    if seed not in DEVELOPMENT_SEEDS:raise DownstreamError('three_development_pipeline_seeds_required')
    source_desc=catalog.get('source_f0',{}).get(str(seed));path=checked(root,source_desc)
    source,receipt=bridge.load_verified_extension_source(root,path.relative_to(root).as_posix())
    if (receipt.get('arm')!=catalog['arm'] or receipt.get('pipeline_seed')!=seed
            or receipt.get('input_contract_sha256')!=INPUT_SHA
            or set(receipt.get('training_dataset_ids',[]))!=set(_datasets(catalog['arm']))):
        raise DownstreamError('same_arm_seed_input_real_extended_f0_required')
    cache=bridge.load_extension_cache(root,arm=catalog['arm'],manifest_path=receipt['extension_manifest']['path'],
        manifest_sha256=receipt['extension_manifest']['sha256'],output=receipt['output'])
    cache.assert_fresh(composite_required=True)
    training=_json(checked(root,receipt['training_uid_manifest']));uids=training.get('event_uids')
    if (training.get('synthetic') is not False or training.get('role')!='source_pretrain_train'
            or training.get('pipeline_seed')!=seed or training.get('signal_contract_sha256')!=INPUT_SHA
            or not isinstance(uids,list) or not uids or len(set(uids))!=len(uids)
            or uids!=[r['event_uid'] for r in cache.rows]):
        raise DownstreamError('real_exact_five_source_f0_training_allowlist_required')
    contract=_json(local(root,'contracts/input_contract.json'))
    if validate_signal_contract(contract)!=INPUT_SHA:raise DownstreamError('frozen_11channel_window_changed')
    return root,catalog,source,receipt,cache,tuple(uids),contract


def _trial_stream(cache,uids):
    index={row['event_uid']:i for i,row in enumerate(cache.rows)}
    for uid in sorted(uids):
        i=index[uid];row=cache.rows[i];shard,trial=cache.locations[i]
        x=torch.from_numpy(np.asarray(cache.arrays[shard][trial]).copy()).to(torch.float64)
        yield SourceCovarianceTrial(row['dataset'],str(row['subject_id']),uid,x,'source_pretrain_train')


def _npy_new(path,array):
    buffer=io.BytesIO();np.save(buffer,np.asarray(array),allow_pickle=False);_bytes_new(path,buffer.getvalue())


def _bytes_new(path,body):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
        try:os.link(temp,path)
        except FileExistsError:
            if path.read_bytes()!=body:raise DownstreamError('immutable_artifact_differs_no_overwrite')
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:Path(temp).unlink(missing_ok=True)


def _owned(root,arm,relative):
    prefix=f'runs/source_extension_downstream/{arm}/'
    if not isinstance(relative,str) or not relative.startswith(prefix):raise DownstreamError('owned_arm_output_required')
    return local(root,relative,exists=False)


def prepare_reference(root,catalog,seed,*,execute=False):
    if execute:_require_owned_stage()
    root,catalog,source,source_receipt,cache,uids,contract=_source_inputs(root,catalog,seed)
    output=_owned(root,catalog['arm'],f"runs/source_extension_downstream/{catalog['arm']}/reference/seed-{seed}/receipt.json")
    identity={'schema_version':SCHEMA,'kind':'source_reference','arm':catalog['arm'],'seed':seed,
        'input_contract_sha256':INPUT_SHA,'source_f0':catalog['source_f0'][str(seed)],
        'combined_cache_sha256':cache.manifest_sha256,'training_UID_set_sha256':_uid_sha(uids),
        'source_dataset_ids':sorted(_datasets(catalog['arm'])),'training_events':len(uids),
        'weighting':'dataset_equal/subject_equal/trial_equal','shrinkage':.05,'code':_code(root),
        'internal_allowed':False,'external_allowed':False}
    if not execute:return {**identity,'status':'PLANNED_NOT_COMPUTED','actual_fits':0,'GPU_checked':False}
    if output.exists():load_reference(root,descriptor(root,output),catalog,seed,source_receipt=source_receipt,cache=cache,uids=uids);return descriptor(root,output)
    reference=source_reference_five(_trial_stream(cache,uids),_datasets(catalog['arm']))
    arrays={}
    for name,value in {'Q':helmert_rows(),'reference':reference.matrix,'sqrt':reference.sqrt,'invsqrt':reference.invsqrt}.items():
        path=output.parent/(name+'.npy');_npy_new(path,value.cpu().numpy());arrays[name]={**descriptor(root,path),'array_sha256':tensor_sha256(value)}
    uid_path=output.parent/'training_UIDs.json';immutable_json(uid_path,{'role':'source_pretrain_train','UIDs':list(reference.event_uids)})
    immutable_json(output,{**identity,'status':'SOURCE_EXTENSION_REFERENCE_COMPLETE','synthetic':False,
        'arrays':arrays,'training_UID_manifest':descriptor(root,uid_path),'actual_neural_fits':0})
    return descriptor(root,output)


def load_reference(root,item,catalog,seed,*,source_receipt=None,cache=None,uids=None):
    root=Path(root).resolve(strict=True);catalog=_catalog(root,catalog);path=checked(root,item);value=_json(path)
    if cache is None:root,catalog,source,source_receipt,cache,uids,contract=_source_inputs(root,catalog,seed)
    checks={'schema_version':SCHEMA,'kind':'source_reference','status':'SOURCE_EXTENSION_REFERENCE_COMPLETE',
        'arm':catalog['arm'],'seed':seed,'input_contract_sha256':INPUT_SHA,'synthetic':False,
        'combined_cache_sha256':cache.manifest_sha256,'training_UID_set_sha256':_uid_sha(uids),
        'source_f0':catalog['source_f0'][str(seed)],'source_dataset_ids':sorted(_datasets(catalog['arm'])),
        'training_events':len(uids),'weighting':'dataset_equal/subject_equal/trial_equal','shrinkage':.05,
        'code':_code(root),'internal_allowed':False,'external_allowed':False}
    if any(value.get(k)!=v for k,v in checks.items()):raise DownstreamError('own_five_source_reference_binding_invalid')
    saved_uids=_json(checked(root,value['training_UID_manifest']))
    if saved_uids!={'role':'source_pretrain_train','UIDs':sorted(uids)}:raise DownstreamError('reference_UID_allowlist_changed')
    shapes={'Q':(10,11),'reference':(10,10),'sqrt':(10,10),'invsqrt':(10,10)};arrays={}
    for name,shape in shapes.items():
        item=value.get('arrays',{}).get(name);array=np.load(checked(root,item),allow_pickle=False)
        if array.shape!=shape or array.dtype!=np.float64 or not np.isfinite(array).all():raise DownstreamError('reference_array_shape_dtype_invalid')
        tensor=torch.from_numpy(array.copy())
        if tensor_sha256(tensor)!=item.get('array_sha256'):raise DownstreamError('reference_numeric_SHA_changed')
        arrays[name]=tensor
    sq,inv=spd_roots(arrays['reference'])
    if (not torch.equal(arrays['Q'],helmert_rows()) or not torch.allclose(arrays['reference'].trace(),torch.tensor(1.,dtype=torch.float64),atol=1e-10,rtol=0)
            or float(torch.linalg.eigvalsh(arrays['reference']).min())<.005-1e-12
            or not torch.allclose(sq,arrays['sqrt'],atol=1e-10,rtol=1e-10) or not torch.allclose(inv,arrays['invsqrt'],atol=1e-10,rtol=1e-10)):
        raise DownstreamError('reference_SPD_roots_or_fixed_Q_invalid')
    return ReferenceCovariance(arrays['reference'],sq,inv,tuple(sorted(uids)),_uid_sha(uids)),value


@dataclass(frozen=True,eq=False)
class ProductionContext(MathContext):
    root: Path
    catalog: dict
    source_receipt: dict
    reference_receipt: dict
    source_cache_sha256: str
    training_UID_set_sha256: str
    code: dict
    issuer: object
    __hash__=object.__hash__
    __eq__=object.__eq__
    def __post_init__(self):
        super().__post_init__()
        if self.issuer is not _ISSUER:raise PermissionError('only_verified_extension_context_factory_can_issue')
    @property
    def production_identity(self):
        return digest({'math':self.identity,'source_f0':self.source_receipt,'reference':self.reference_receipt,
            'source_cache':self.source_cache_sha256,'training_UIDs':self.training_UID_set_sha256,'code':self.code})
    def assert_current(self):
        seal=_CONTEXT_SEALS.get(self)
        if self.issuer is not _ISSUER or seal is None:raise PermissionError('unissued_production_context')
        if (self.source is not seal['source'] or self.reference is not seal['reference']
                or self.identity!=seal['math_identity'] or self.production_identity!=seal['production_identity']
                or _reference_numeric(self.reference)!=seal['reference_arrays']):
            raise DownstreamError('issued_source_interface_reference_or_context_changed')
        checked(self.root,self.source_receipt);checked(self.root,self.reference_receipt)
        if _code(self.root)!=self.code:raise DownstreamError('downstream_code_changed_after_context_issue')
        self.source.assert_unchanged()


def _reference_numeric(reference):
    return {name:{'sha256':tensor_sha256(getattr(reference,name)),
                  'shape':list(getattr(reference,name).shape),
                  'stride':list(getattr(reference,name).stride()),
                  'dtype':str(getattr(reference,name).dtype),'layout':str(getattr(reference,name).layout)}
            for name in ('matrix','sqrt','invsqrt')}


def _seal_context(context):
    if type(context) is not ProductionContext or context.issuer is not _ISSUER:
        raise PermissionError('only_verified_factory_can_seal_context')
    _CONTEXT_SEALS[context]={'source':context.source,'reference':context.reference,
        'math_identity':context.identity,'production_identity':context.production_identity,
        'reference_arrays':_reference_numeric(context.reference)}
    return context


def load_context(root,catalog,seed=SEEDS[0]):
    root,catalog,source,receipt,cache,uids,contract=_source_inputs(root,catalog,seed)
    f0_stage=verify_f0_stage(root,descriptor(root,_owned(root,catalog['arm'],f"runs/source_extension_downstream/{catalog['arm']}/source_f0/seed-{seed}/receipt.json")),catalog)
    if f0_stage['source_f0']!=catalog['source_f0'][str(seed)]:raise DownstreamError('own_bounded_source_f0_stage_required')
    ref_desc=catalog.get('references',{}).get(str(seed))
    reference_stage=verify_reference_stage(root,descriptor(root,_owned(root,catalog['arm'],f"runs/source_extension_downstream/{catalog['arm']}/reference_stage/seed-{seed}/receipt.json")),catalog)
    if reference_stage['reference']!=ref_desc:raise DownstreamError('own_bounded_reference_stage_required')
    reference,_=load_reference(root,ref_desc,catalog,seed,source_receipt=receipt,cache=cache,uids=uids)
    train=load_episode_catalog(root,role='meta_train',purpose='meta_gradient')
    val=load_episode_catalog(root,role='meta_validation',purpose='model_selection')
    if len(train.partition_people)!=40 or len(val.partition_people)!=11 or set(train.partition_people)&set(val.partition_people) or train.split_sha256!=val.split_sha256:
        raise DownstreamError('real_frozen_40train_11validation_required')
    if train.input_contract_sha256!=INPUT_SHA or val.input_contract_sha256!=INPUT_SHA:
        raise DownstreamError('development_input_window_mismatch')
    context=ProductionContext(source,reference,catalog['arm'],seed,INPUT_SHA,train.split_sha256,root,catalog,
        catalog['source_f0'][str(seed)],ref_desc,cache.manifest_sha256,_uid_sha(uids),_code(root),_ISSUER)
    return _seal_context(context),train,val


def _production(ctx):
    if type(ctx) is not ProductionContext or ctx.issuer is not _ISSUER:raise PermissionError('real_verified_extension_context_required')
    ctx.assert_current()


def _to_saved(value):
    if isinstance(value,np.ndarray):return torch.from_numpy(value.copy())
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:_to_saved(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_to_saved(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value


def _adapter_payload(state):
    return {'schema_version':SCHEMA,'kind':'support_fit_payload','method':state.method,'configuration':state.configuration,
        'context_sha256':state.context_sha256,'payload':_to_saved(state.payload),'dictionary':_to_saved(state.dictionary),
        'updates':state.updates,'fallback_reason':state.fallback_reason,'trainable_parameters':state.trainable_parameters,
        'support_event_uids':list(state.support_event_uids),'acquired_event_uids':list(state.acquired_event_uids),'state_sha256':state.state_sha256}


def save_adapter(ctx,state,folder,candidate):
    _production(ctx)
    entry=_FIT_SEALS.get(id(state))
    if (entry is None or entry[0]() is not state or entry[1]!=digest(candidate)
            or entry[2]!=ctx.production_identity):
        raise PermissionError('mathematical_payload_is_not_a_production_dispatched_fit')
    actual=_adapter_sha(state.method,state.configuration,state.context_sha256,state.payload,state.dictionary,
        state.updates,state.fallback_reason,state.trainable_parameters,state.support_event_uids,state.acquired_event_uids)
    if actual!=state.state_sha256 or actual!=entry[3]:raise DownstreamError('issued_fit_parameters_changed_before_commit')
    if state.context_sha256!=ctx.identity or candidate.get('context')!=ctx.production_identity or candidate.get('arm')!=ctx.arm:
        raise DownstreamError('candidate_or_fit_context_mismatch')
    folder=Path(folder)
    try:relative=folder.relative_to(ctx.root).as_posix()
    except ValueError:raise DownstreamError('owned_arm_output_required') from None
    folder=_owned(ctx.root,ctx.arm,relative);folder.mkdir(parents=True,exist_ok=True)
    immutable_json(folder/'candidate.json',candidate)
    payload={**_adapter_payload(state),'production_context_identity':ctx.production_identity,'candidate_sha256':digest(candidate)}
    path=folder/'fit.pt'
    if path.exists():
        if not _same_tree(verified_checkpoint(path),payload):raise DownstreamError('immutable_fit_payload_changed')
    else:atomic_checkpoint(path,payload)
    record={'schema_version':SCHEMA,'status':'SOURCE_EXTENSION_FIT_PAYLOAD_COMMITTED','arm':ctx.arm,
        'context':ctx.production_identity,'candidate':descriptor(ctx.root,folder/'candidate.json'),
        'checkpoint':descriptor(ctx.root,path),'integrity':descriptor(ctx.root,checkpoint_sidecar(path)),
        'state_sha256':state.state_sha256,'scientific_meta_fit_claim':False}
    immutable_json(folder/'fit_receipt.json',record)
    return descriptor(ctx.root,folder/'fit_receipt.json')


def _same_tree(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(_same_tree(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return isinstance(b,(tuple,list)) and len(a)==len(b) and all(_same_tree(x,y) for x,y in zip(a,b))
    return a==b


def load_adapter(ctx,item,candidate):
    _production(ctx);value=_json(checked(ctx.root,item))
    if value.get('status')!='SOURCE_EXTENSION_FIT_PAYLOAD_COMMITTED' or value.get('context')!=ctx.production_identity:
        raise DownstreamError('same_arm_committed_fit_required')
    if _json(checked(ctx.root,value['candidate']))!=candidate:raise DownstreamError('fit_candidate_changed')
    cp=checked(ctx.root,value['checkpoint']);checked(ctx.root,value['integrity']);payload=verified_checkpoint(cp)
    if payload.get('candidate_sha256')!=digest(candidate) or payload.get('production_context_identity')!=ctx.production_identity:
        raise DownstreamError('fitted_payload_context_or_candidate_changed')
    arrays=payload['payload']
    if payload['method'] in {'B4','B5'}:arrays['arrays']={k:v.detach().cpu().numpy().copy() for k,v in arrays['arrays'].items()}
    state=AdapterState(payload['method'],payload['configuration'],payload['context_sha256'],arrays,payload['dictionary'],payload['updates'],payload['fallback_reason'],payload['trainable_parameters'],tuple(payload['support_event_uids']),tuple(payload['acquired_event_uids']),payload['state_sha256'])
    # Move numerical payloads without changing dtype, strides or bound identity.
    device=next(ctx.source.parameters()).device
    if state.method not in {'B4','B5'}:
        def move(x):return x.to(device) if isinstance(x,torch.Tensor) else {k:move(v) for k,v in x.items()} if isinstance(x,dict) else x
        state=AdapterState(state.method,state.configuration,state.context_sha256,move(state.payload),state.dictionary.to(device) if state.dictionary is not None else None,state.updates,state.fallback_reason,state.trainable_parameters,state.support_event_uids,state.acquired_event_uids,state.state_sha256)
    expected=_adapter_sha(state.method,state.configuration,state.context_sha256,state.payload,state.dictionary,state.updates,state.fallback_reason,state.trainable_parameters,state.support_event_uids,state.acquired_event_uids)
    if expected!=value['state_sha256'] or expected!=state.state_sha256:raise DownstreamError('fitted_numeric_payload_SHA_changed')
    return state


def initialize_catalog(root,arm,primary_freeze,extension_manifest,*,execute=False):
    """Explicit immutable launch specification. No numeric EEG or CUDA read."""
    _datasets(arm);root=Path(root).resolve(strict=True)
    def ref(value):
        return descriptor(root,checked(root,value)) if isinstance(value,dict) else descriptor(root,local(root,str(value)))
    value={'schema_version':SCHEMA,'arm':arm,'input_contract_sha256':INPUT_SHA,
        'primary_development_freeze':ref(primary_freeze),'extension_manifest':ref(extension_manifest),
        'source_f0':{},'references':{},'source_banks':{},'meta':{},'meta_engineering':{},
        'meta_selections':{},'configurations':{},'validation':{},'profiles':{},
        'internal_allowed':False,'external_allowed':False,'primary_H1_changed':False}
    _primary(root,value)
    if execute:
        path=_owned(root,arm,f'runs/source_extension_downstream/{arm}/launch.json')
        immutable_json(path,value);return descriptor(root,path)
    return value


def _catalog_path(root,catalog):
    return _owned(root,catalog['arm'],f"runs/source_extension_downstream/{catalog['arm']}/catalog.json")


def _update_catalog(root,catalog,section,key,item):
    """Append a reference only; an existing different artifact is never replaced."""
    checked(root,item);target=_catalog_path(root,catalog)
    current=_catalog(root,target.relative_to(root)) if target.exists() else copy.deepcopy(catalog)
    if current['primary_development_freeze']!=catalog['primary_development_freeze'] or current['extension_manifest']!=catalog['extension_manifest']:
        raise DownstreamError('launch_identity_changed')
    prior=current.setdefault(section,{}).get(str(key))
    if prior is not None and prior!=item:raise DownstreamError('committed_stage_reference_conflict')
    current[section][str(key)]=item;atomic_json(target,current)
    return current


def _latest_catalog(root,catalog):
    value=_catalog(root,catalog);path=_catalog_path(root,value)
    if not path.exists():return value
    actual=_catalog(root,path.relative_to(root))
    for key in ('arm','primary_development_freeze','extension_manifest','input_contract_sha256'):
        if actual.get(key)!=value.get(key):raise DownstreamError('launch_identity_changed')
    return actual


def _catalog_binding(catalog):
    return {'role':catalog.role,'split_sha256':catalog.split_sha256,
        'manifest_sha256':catalog.manifest_sha256,'input_contract_sha256':catalog.input_contract_sha256,
        'people':list(catalog.partition_people),'plans':[p.to_dict() for p in catalog.plans]}


def _primary_seed(frozen,seed):
    refs=[r['dependencies']['source_f0'] for r in frozen['selected_jobs'] if r['seed']==seed]
    if not refs or len({digest(r) for r in refs})!=1:raise DownstreamError('primary_same_seed_f0_not_unique')
    return refs[0]


def _primary_measurements(root,catalog):
    """Use actual verified baseline measurements only, never invented seconds."""
    from .development_evaluation import verify_calibration_profile
    from .meta_engineering import verify_meta_engineering
    from . import source_extension_training as extension
    from .source_training import verify_engineering_pilot
    primary=_primary(root,catalog);manifest=catalog['extension_manifest'];checked(root,manifest)
    plans={};cache=None
    for seed in DEVELOPMENT_SEEDS:
        cache,plan=extension.build_plan(root,arm=catalog['arm'],seed=seed,
            extension_manifest=manifest['path'],extension_sha256=manifest['sha256'],
            baseline_receipt=_primary_seed(primary,seed)['path'])
        plans[str(seed)]=plan
    baseline=_json(checked(root,_primary_seed(primary,SEEDS[0])))
    pilot_ref=baseline.get('engineering_pilot')
    if not isinstance(pilot_ref,dict):raise DownstreamError('actual_baseline_source200_receipt_required')
    pilot_ref=descriptor(root,checked(root,pilot_ref.get('receipt')))
    verify_engineering_pilot(cache.baseline,pilot_ref['path'])
    pilot=_json(checked(root,pilot_ref));source_seconds=pilot.get('total_monotonic_wall_seconds')
    if not _positive(source_seconds):raise DownstreamError('source200_measured_elapsed_required')
    profiles={};meta={};maxima={};runtime=None
    for selected in primary['selected_jobs']:
        ref=selected['profile_receipt']
        if digest(ref) not in profiles:
            verified=verify_calibration_profile(root,ref)
            if runtime is not None and runtime!=verified.runtime_profile:raise DownstreamError('primary_profiles_same_device_required')
            runtime=verified.runtime_profile;profiles[digest(ref)]=ref
            maxima.update({k:max(maxima.get(k,0),v) for k,v in verified.maximum_calibration_seconds.items()})
        method=selected['method']
        if method in LEARNED and method not in meta and selected['seed']==SEEDS[0]:
            learned=_json(checked(root,selected['dependencies']['learned']))
            budget=_json(checked(root,learned['operational_budget']))
            engineer=budget.get('measured_meta_engineering_receipt')
            proof=verify_meta_engineering(root,engineer,job=MetaJob(**learned['job']),inputs=learned)
            if not _positive(proof.get('active_wall_seconds')):raise DownstreamError('actual_meta100_profile_seconds_required')
            meta[method]={'receipt':descriptor(root,checked(root,engineer)),
                'seconds_for_100':proof['active_wall_seconds']}
    if set(meta)!=LEARNED or not maxima:raise DownstreamError('full_primary_engineering_and_calibration_profiles_required')
    return primary,cache,plans,pilot_ref,float(source_seconds),profiles,meta,maxima,runtime


def prepare_budget(root,catalog,*,execute=False):
    """Fixed total cap projected from actual primary profiles; no fit or CUDA."""
    root=Path(root).resolve(strict=True);catalog=_latest_catalog(root,catalog)
    primary,cache,plans,pilot,source_seconds,profiles,meta,maxima,runtime=_primary_measurements(root,catalog)
    val=load_episode_catalog(root,role='meta_validation',purpose='model_selection')
    train=load_episode_catalog(root,role='meta_train',purpose='meta_gradient')
    if len(train.partition_people)!=40 or len(val.partition_people)!=11:raise DownstreamError('real_40_11_partition_required')
    # These are conservative fixed *projections*, not measured extension times.
    # The source-only CPU projection explicitly scales actual target timing by
    # source trial count/20; a slower new arm pauses rather than inflating caps.
    f0={str(s):math.ceil(SAFETY_FACTOR*source_seconds*plans[str(s)]['optimizer_updates']/200) for s in DEVELOPMENT_SEEDS}
    target_max=max(maxima.values());eligible=sum(p.query_eligible for p in val.plans)
    if not _positive(target_max) or eligible<11:raise DownstreamError('actual_complete_primary_timing_and_val_coverage_required')
    caps={'f0':f0,'reference':SAFETY_FACTOR*source_seconds*len(cache.rows)/max(1,len(cache.baseline.rows)),
        'source_bank_per_job':SAFETY_FACTOR*max(source_seconds,target_max*len(cache.rows)/20),
        'meta_engineering':{m:SAFETY_FACTOR*meta[m]['seconds_for_100'] for m in sorted(LEARNED)},
        'meta_science_per_job':{m:SAFETY_FACTOR*meta[m]['seconds_for_100']*10 for m in sorted(LEARNED)},
        'calibration_profile':SAFETY_FACTOR*sum(maxima.values())*118,
        'validation_per_job':SAFETY_FACTOR*target_max*eligible*2}
    total=(sum(f0.values())+3*caps['reference']+6*caps['source_bank_per_job']
        +sum(caps['meta_engineering'].values())+11*sum(caps['meta_science_per_job'].values())
        +3*caps['calibration_profile']+146*caps['validation_per_job'])
    value={'schema_version':SCHEMA,'status':'FIXED_PROFILE_DERIVED_EXECUTION_BUDGET',
        'arm':catalog['arm'],'input_contract_sha256':INPUT_SHA,'primary_freeze':catalog['primary_development_freeze'],
        'extension_manifest':catalog['extension_manifest'],'source_plans':plans,'source_pilot':pilot,
        'source200_measured_seconds':source_seconds,'primary_profiles':list(profiles.values()),
        'primary_meta_engineering':meta,'primary_calibration_maxima':maxima,'runtime':runtime,
        'training_UID_set_sha256':_uid_sha([r['event_uid'] for r in cache.rows]),
        'combined_cache_sha256':cache.manifest_sha256,'source_trials':len(cache.rows),
        'development_catalog':_catalog_binding(val),'meta_catalog':_catalog_binding(train),
        'safety_factor':SAFETY_FACTOR,'projection_not_extension_measurement':True,
        'caps':caps,'max_total_wall_seconds':total,'max_scientific_meta_jobs':55,
        'max_engineering_meta_jobs':5,'max_observed_outer_calls':111000,
        'max_source_f0_optimizer_calls':2*sum(p['optimizer_updates'] for p in plans.values()),
        'max_source_bank_fits':6,'max_target_fits':3*118+146*eligible,
        'max_download_bytes':0,'internal_allowed':False,'external_allowed':False,'code':_code(root)}
    if not _positive(total):raise DownstreamError('finite_profile_derived_total_cap_required')
    if execute:
        path=_owned(root,catalog['arm'],f"runs/source_extension_downstream/{catalog['arm']}/budget.json")
        immutable_json(path,value);return descriptor(root,path)
    return value


def _budget(root,catalog):
    path=_owned(root,catalog['arm'],f"runs/source_extension_downstream/{catalog['arm']}/budget.json")
    if not path.exists():raise DownstreamError('profile_derived_fixed_arm_budget_required')
    value=_json(path)
    if value!=prepare_budget(root,catalog):raise DownstreamError('budget_profile_data_code_or_fixed_caps_changed')
    return value,descriptor(root,path)


def _closed_wall(value,identity=None,maximum=None):
    """Read-only final gate, including work after the last optimizer/predict."""
    charged=value.get('durable_wall_seconds');limit=value.get('max_wall_seconds')
    if (value.get('interval') is not None or value.get('clock_block') is not None
            or type(charged) not in (int,float) or not math.isfinite(charged) or charged<0
            or not _positive(limit) or charged>limit or value.get('amendments')!=[]
            or identity is not None and value.get('identity_sha256')!=identity
            or maximum is not None and limit!=maximum):
        raise DownstreamError('closed_resource_wall_invalid_or_overrun_no_completion')
    return charged


class ArmRun:
    """A single arm owner and cumulative ledger, retained across all stages.

    Resumable meta/f0 trajectories retain explicit physical call counts. Target
    fits/CPU solvers have one dispatch each: unknown outcomes cannot be retried.
    """
    def __init__(self,root,catalog,runtime):
        self.root=Path(root).resolve(strict=True);self.catalog=catalog;self.arm=catalog['arm'];self.runtime=runtime
        self.budget,self.budget_ref=_budget(self.root,catalog)
        if runtime!=self.budget.get('runtime'):raise DownstreamError('actual_primary_profile_same_device_runtime_required')
        self.folder=_owned(self.root,self.arm,f'runs/source_extension_downstream/{self.arm}/resources')
        self.folder.mkdir(parents=True,exist_ok=True);self.lock=(self.folder/'owner.lock').open('a')
        try:fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:self.lock.close();raise DownstreamError('single_extension_arm_owner_already_active') from None
        self.identity={'schema_version':SCHEMA,'arm':self.arm,'budget':self.budget_ref,'runtime':runtime}
        self.identity_sha=digest(self.identity);self.path=self.folder/'ledger.json';self.wall=None
        try:
            old_claim=(self.folder/'claim.json').exists()
            if old_claim and (not self.path.exists() or not (self.folder/'wall.json').exists()):
                raise DownstreamError('issued_arm_missing_ledger_or_wall_no_reset')
            immutable_json(self.folder/'claim.json',self.identity)
            if self.path.exists():
                self.state=_json(self.path)
                if self.state.get('identity_sha256')!=self.identity_sha:raise DownstreamError('arm_resource_identity_changed')
            else:
                if any(self.folder.glob('*.pt')) or (self.folder/'wall.json').exists():raise DownstreamError('lost_arm_ledger_cannot_be_reset')
                self.state={'identity_sha256':self.identity_sha,'target_fits':{},'bank_fits':{},'meta_calls':{},'source_calls':{},'stages':{},'wall_high_water_seconds':0.}
                self.save()
            _validate_ledger(self.state)
            floor=self.state.get('wall_high_water_seconds')
            if type(floor) not in (int,float) or not math.isfinite(floor) or floor<0:raise DownstreamError('arm_wall_high_water_invalid')
            for snapshot in (self.folder/'snapshots').glob('*.json'):
                value=_json(snapshot)
                if snapshot.stem!=digest(value) or value.get('identity_sha256')!=self.identity_sha:
                    raise DownstreamError('arm_immutable_resource_snapshot_changed')
                floor=max(floor,_closed_wall(value['closed_wall'],self.identity_sha,self.budget['max_total_wall_seconds']))
                _ledger_prefix(value['ledger'],self.state)
            for claim_path in (self.folder/'dispatches').glob('*/*.json'):
                record=_json(claim_path);kind=claim_path.parent.name;key=claim_path.stem
                if record.get('identity_sha256')!=self.identity_sha or self.state.get(kind,{}).get(key)!=record.get('claim'):
                    raise DownstreamError('immutable_dispatch_claim_missing_from_live_ledger')
            for count_path in (self.folder/'calls').glob('*/*.json'):
                record=_json(count_path);kind=count_path.parent.name
                if (record.get('identity_sha256')!=self.identity_sha or type(record.get('count')) is not int
                        or record['count']<0 or self.state.get(kind,{}).get(record.get('key'),-1)<record['count']):
                    raise DownstreamError('immutable_physical_call_high_water_missing_from_live_ledger')
            self.wall=PersistentWallBudget(self.folder/'wall.json',self.identity_sha,self.budget['max_total_wall_seconds'],checkpoint_wall_seconds=floor)
            self.wall.start_interval();self.guard()
        except BaseException:
            if self.wall is not None:self.wall.close()
            self.lock.close();raise
    def save(self):
        if self.wall is not None:self.state['wall_high_water_seconds']=max(self.state.get('wall_high_water_seconds',0),self.wall.wall_seconds)
        atomic_json(self.path,self.state)
    def guard(self):
        _validate_ledger(self.state)
        if self.wall.exhausted:raise DownstreamError('fixed_shared_arm_wall_budget_exhausted')
        for field,limit in (('target_fits','max_target_fits'),('bank_fits','max_source_bank_fits')):
            if len(self.state[field])>self.budget[limit]:raise DownstreamError('shared_arm_fit_call_cap_exceeded')
        for field,limit in (('meta_calls','max_observed_outer_calls'),('source_calls','max_source_f0_optimizer_calls')):
            if sum(self.state[field].values())>self.budget[limit]:raise DownstreamError('shared_arm_optimizer_call_cap_exceeded')
        self.wall.checkpoint();self.save()
    def dispatch_once(self,kind,key,identity,committed=False):
        self.guard();table=self.state[kind];claim={'identity_sha256':digest(identity),'dispatched':True}
        if key in table:
            if table[key]!=claim:raise DownstreamError('dispatch_identity_changed')
            if not committed:raise DownstreamError('unknown_dispatched_fit_no_commit_refit_forbidden')
            return False
        if committed:raise DownstreamError('committed_fit_missing_dispatch_ledger')
        immutable_json(self.folder/'dispatches'/kind/(key+'.json'),{'identity_sha256':self.identity_sha,'claim':claim})
        table[key]=claim;self.save();self.guard();return True
    def observe(self,kind,key,count):
        if type(count) is not int or count<0:raise DownstreamError('nonnegative_physical_call_count_required')
        previous=self.state[kind].get(key,0)
        if count<previous:raise DownstreamError('physical_call_high_water_moved_backwards')
        immutable_json(self.folder/'calls'/kind/(digest({'key':key,'count':count})+'.json'),
            {'identity_sha256':self.identity_sha,'key':key,'count':count})
        self.state[kind][key]=count;self.save();self.guard()
    def close(self):
        if self.wall is None:return
        try:
            self.wall.graceful_close();self.save();self.wall.close();value=_json(self.folder/'wall.json')
            _closed_wall(value,self.identity_sha,self.budget['max_total_wall_seconds'])
            snapshot={'identity_sha256':self.identity_sha,'closed_wall':value,'ledger':copy.deepcopy(self.state)}
            immutable_json(self.folder/'snapshots'/(digest(snapshot)+'.json'),snapshot)
        finally:self.wall=None;self.lock.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def snapshot(self,folder):
        if self.wall is not None:raise DownstreamError('resources_must_close_before_completion_snapshot')
        wall=_json(self.folder/'wall.json');_closed_wall(wall,self.identity_sha,self.budget['max_total_wall_seconds'])
        value={'schema_version':SCHEMA,'claim':descriptor(self.root,self.folder/'claim.json'),
            'budget':self.budget_ref,'closed_wall':wall,'ledger':copy.deepcopy(self.state),
            'live_wall_sha256':file_digest(self.folder/'wall.json'),'live_ledger_sha256':file_digest(self.path)}
        path=Path(folder)/'resource_snapshot.json';immutable_json(path,value);return descriptor(self.root,path)


def _resource_proof(root,item,arm):
    value=_json(checked(root,item));claim=_json(checked(root,value['claim']));budget=_json(checked(root,value['budget']))
    if claim.get('arm')!=arm or claim.get('budget')!=value['budget'] or budget.get('arm')!=arm:
        raise DownstreamError('resource_snapshot_arm_or_budget_changed')
    identity=digest(claim);_closed_wall(value['closed_wall'],identity,budget['max_total_wall_seconds'])
    ledger=value.get('ledger',{})
    _validate_ledger(ledger)
    if ledger.get('identity_sha256')!=identity:raise DownstreamError('resource_snapshot_ledger_identity_changed')
    for key,limit in (('target_fits','max_target_fits'),('bank_fits','max_source_bank_fits')):
        if len(ledger.get(key,{}))>budget[limit]:raise DownstreamError('resource_snapshot_call_limit_exceeded')
    for key,limit in (('meta_calls','max_observed_outer_calls'),('source_calls','max_source_f0_optimizer_calls')):
        if sum(ledger.get(key,{}).values())>budget[limit]:raise DownstreamError('resource_snapshot_optimizer_cap_exceeded')
    return value,claim,budget


def _validate_ledger(ledger):
    if not isinstance(ledger,dict):raise DownstreamError('typed_resource_ledger_required')
    for field in ('target_fits','bank_fits','stages','meta_calls','source_calls'):
        if not isinstance(ledger.get(field),dict):raise DownstreamError('complete_resource_ledger_tables_required')
    for field in ('meta_calls','source_calls'):
        if any(type(v) is not int or v<0 for v in ledger[field].values()):raise DownstreamError('nonnegative_integer_physical_calls_required')
    for field in ('target_fits','bank_fits'):
        for value in ledger[field].values():
            if set(value)!={'identity_sha256','dispatched'} or value['dispatched'] is not True:
                raise DownstreamError('immutable_numerical_dispatch_claim_required')
            _sha(value['identity_sha256'])


def _ledger_prefix(old,current):
    _validate_ledger(old);_validate_ledger(current)
    if old.get('identity_sha256')!=current.get('identity_sha256'):raise DownstreamError('resource_ledger_identity_changed')
    for field in ('target_fits','bank_fits'):
        if any(current[field].get(k)!=v for k,v in old[field].items()):raise DownstreamError('committed_dispatch_ledger_rolled_back')
    for field in ('meta_calls','source_calls'):
        if any(current[field].get(k,-1)<v for k,v in old[field].items()):raise DownstreamError('physical_call_ledger_rolled_back')
    for key,stage in old['stages'].items():
        latest=current['stages'].get(key)
        if not isinstance(latest,dict) or any(latest.get(k)!=v for k,v in stage.items() if k!='wall_high_water_seconds'):
            raise DownstreamError('stage_dispatch_ledger_rolled_back')
        if latest.get('wall_high_water_seconds',0)<stage.get('wall_high_water_seconds',0):raise DownstreamError('stage_wall_high_water_rolled_back')


@contextmanager
def _stage_wall(run,folder,key,identity,maximum):
    """Durable per-stage cap in addition to the cumulative arm cap."""
    path=Path(folder)/'stage_wall.json';sha=digest(identity)
    old=run.state['stages'].get(key)
    if old is not None and (old.get('identity_sha256')!=sha or not path.is_file()):
        raise DownstreamError('stage_identity_or_durable_wall_missing_no_reset')
    if old is None and path.exists():raise DownstreamError('stage_wall_without_dispatch_claim')
    floor=old.get('wall_high_water_seconds',0) if old else 0
    run.state['stages'][key]={'identity_sha256':sha,'kind':'bounded_stage','dispatched':True,'wall_high_water_seconds':floor}
    run.save();wall=PersistentWallBudget(path,sha,maximum,checkpoint_wall_seconds=floor)
    try:
        wall.start_interval()
        if wall.exhausted:raise DownstreamError('stage_cumulative_wall_exhausted')
        yield wall
    finally:
        try:
            wall.graceful_close()
            run.state['stages'][key]['wall_high_water_seconds']=max(floor,wall.wall_seconds)
            run.save()
        finally:wall.close()


def _verify_stage_wall(root,item,identity,maximum,resources,key):
    value=_json(checked(root,item));elapsed=_closed_wall(value,digest(identity),maximum)
    stage=resources['ledger'].get('stages',{}).get(key)
    if (not isinstance(stage,dict) or stage.get('identity_sha256')!=digest(identity)
            or stage.get('kind')!='bounded_stage' or stage.get('dispatched') is not True
            or stage.get('wall_high_water_seconds')!=elapsed):
        raise DownstreamError('stage_wall_dispatch_and_high_water_required')
    return elapsed


def _runtime():
    # This helper is reached only after --execute AND --user-started-gpu gates.
    from .development_evaluation import _production_runtime
    with threadpool_limits(limits=1):return _production_runtime()


def _require_owned_stage():
    if _OWNED_STAGE is not True or os.getpid()!=os.getpgrp():
        raise DownstreamError('bounded_stage_runner_required')


def _gpu(ctx):
    ctx.source.to('cuda:0');ctx.source.assert_unchanged()
    if any(p.device.type!='cuda' for p in ctx.source.parameters()):raise DownstreamError('actual_CUDA_source_required_no_CPU_fit')


def _deadline(seconds):
    @contextmanager
    def bounded():
        if not _positive(seconds):raise DownstreamError('positive_bounded_fit_deadline_required')
        if signal.getsignal(signal.SIGALRM)!=signal.SIG_DFL:raise DownstreamError('dedicated_process_deadline_owner_required')
        def expired(*_):raise DownstreamError('candidate_deadline_expired_no_completion')
        previous=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,seconds)
        try:yield
        finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    return bounded()


def _source_bank_worker(x_path,y,rows,method,configuration,ids,connection,owner_pid,deadline):
    """Finite isolated CPU solver; parent always kills its process group on exit."""
    try:
        os.setsid()
        from .training_campaign import start_stage_watchdog
        stop,watchdog=start_stage_watchdog(owner_pid,os.getpgrp(),deadline)
        with threadpool_limits(limits=1):
            bank=fit_source_bank_math(np.load(x_path,mmap_mode='r',allow_pickle=False),y,rows,method,configuration,ids,INPUT_SHA)
        connection.send(('ok',bank))
    except BaseException as exc:connection.send(('error',type(exc).__name__))
    finally:
        if 'stop' in locals():stop.set();watchdog.join(.2)
        connection.close()


def _solve_bank(x_path,cache,method,configuration,seconds):
    parent,child=multiprocessing.get_context('spawn').Pipe(duplex=False)
    deadline=time.monotonic()+seconds
    worker=multiprocessing.get_context('spawn').Process(target=_source_bank_worker,
        args=(str(x_path),np.asarray(cache.labels),cache.rows,method,configuration,_datasets(cache.arm),child,os.getpid(),deadline))
    started=time.monotonic();worker.start();child.close()
    try:
        if not parent.poll(seconds):raise DownstreamError('source_bank_CPU_deadline_expired')
        status,result=parent.recv()
        if status!='ok':raise DownstreamError('source_bank_CPU_solver_failed_'+result)
        worker.join(max(0,seconds-(time.monotonic()-started)))
        if worker.is_alive() or worker.exitcode!=0:raise DownstreamError('source_bank_worker_not_cleanly_joined')
        if time.monotonic()-started>seconds:raise DownstreamError('source_bank_CPU_finalization_deadline_expired')
        return result,time.monotonic()-started
    finally:
        if worker.is_alive():
            try:os.killpg(worker.pid,signal.SIGKILL)
            except ProcessLookupError:worker.kill()
            worker.join()
        parent.close()


def _bank_key(method,configuration):return method+'-'+digest(configuration)[:16]


def source_bank_configurations():
    return [('B4',{'m':m}) for m in (1,2,3)]+[('B5',{'shrinkage':v}) for v in (.01,.05,.1)]


@threadpool_limits.wrap(limits=1)
def prepare_source_banks(root,catalog,*,execute=False,user_started_gpu=False):
    if not execute:return {**build_plan(_catalog(root,catalog)['arm']),'stage':'source-bank','status':'PLANNED_NOT_EXECUTED'}
    if not user_started_gpu:raise PermissionError('execute_and_later_user_started_GPU_both_required')
    _require_owned_stage()
    root=Path(root).resolve(strict=True);catalog=_latest_catalog(root,catalog)
    root,catalog,source,sreceipt,cache,uids,contract=_source_inputs(root,catalog,SEEDS[0]);runtime=_runtime()
    folder=_owned(root,catalog['arm'],f"runs/source_extension_downstream/{catalog['arm']}/source_banks")
    folder.mkdir(parents=True,exist_ok=True);pending=[];new_fits=0
    run=ArmRun(root,catalog,runtime)
    try:
        # Never make an old standardized NPZ or silently swap units: cache.batch
        # is the exact verified pre-RMS clinic11 native crop used by new f0.
        raw=folder/'source_trials.npy'
        if not raw.exists():
            temp=raw.with_suffix('.partial.npy')
            if temp.exists():raise DownstreamError('uncommitted_source_bank_input_preserved')
            x=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float32,shape=(len(cache.rows),11,128))
            for start in range(0,len(cache.rows),128):
                run.guard();x[start:start+128]=cache.batch(np.arange(start,min(start+128,len(cache.rows))))
            x.flush();del x;os.link(temp,raw);temp.unlink()
        raw_array=np.load(raw,mmap_mode='r',allow_pickle=False)
        if raw_array.shape!=(len(cache.rows),11,128) or raw_array.dtype!=np.float32:raise DownstreamError('exact_source_bank_native_cache_shape_dtype_required')
        for start in range(0,len(cache.rows),128):
            run.guard()
            if not np.array_equal(raw_array[start:start+128],cache.batch(np.arange(start,min(start+128,len(cache.rows))))):
                raise DownstreamError('source_bank_input_differs_from_actual_verified_cache')
        del raw_array;raw_desc=descriptor(root,raw)
        for method,configs in [('B4',[{'m':m} for m in (1,2,3)]),('B5',[{'shrinkage':v} for v in (.01,.05,.1)])]:
            for configuration in configs:
                key=_bank_key(method,configuration);dest=folder/key;dest.mkdir(exist_ok=True)
                identity={'schema_version':SCHEMA,'kind':'five_source_bank','arm':catalog['arm'],'method':method,
                    'configuration':configuration,'input_contract_sha256':INPUT_SHA,'source_dataset_ids':sorted(_datasets(catalog['arm'])),
                    'cache_sha256':cache.manifest_sha256,'UID_set_sha256':_uid_sha(uids),'source_input':raw_desc,
                    'code':_code(root),'runtime':runtime,'budget':run.budget_ref}
                committed=(dest/'payload.json').exists()
                dispatch=run.dispatch_once('bank_fits',key,identity,committed)
                immutable_json(dest/'identity.json',identity)
                if dispatch:
                    bank,elapsed=_solve_bank(raw,cache,method,configuration,min(run.wall.remaining_seconds,run.budget['caps']['source_bank_per_job']))
                    new_fits+=1
                    if not bank.metadata.get('converged'):raise DownstreamError('source_bank_nonconverged_preserved_no_complete')
                    atomic_checkpoint(dest/'bank.pt',{'method':method,'arrays':_to_saved(bank.arrays),'metadata':bank.metadata,'state_sha256':bank.state_sha256})
                    immutable_json(dest/'payload.json',{'identity':descriptor(root,dest/'identity.json'),'checkpoint':descriptor(root,dest/'bank.pt'),
                        'sidecar':descriptor(root,checkpoint_sidecar(dest/'bank.pt')),'elapsed_seconds':elapsed})
                pending.append((key,dest,method,configuration))
                run.guard()
    finally:run.close()
    ctx,_,_=load_context(root,catalog)
    for key,dest,method,configuration in pending:
        receipt=dest/'receipt.json'
        if not receipt.exists():
            resources=run.snapshot(dest)
            immutable_json(receipt,{'schema_version':SCHEMA,'status':'SOURCE_EXTENSION_CLASSICAL_BANK_COMPLETE','synthetic':False,
                'arm':catalog['arm'],'payload':descriptor(root,dest/'payload.json'),'resources':resources,
                'internal_allowed':False,'external_allowed':False})
        load_source_bank(ctx,descriptor(root,receipt),method,configuration)
        catalog=_update_catalog(root,catalog,'source_banks',key,descriptor(root,receipt))
    return {'status':'SOURCE_EXTENSION_SIX_SOURCE_BANKS_COMPLETE','catalog':descriptor(root,_catalog_path(root,catalog)),'actual_new_fits':new_fits}


def load_source_bank(ctx,item,method,configuration):
    _production(ctx);value=_json(checked(ctx.root,item))
    if value.get('status')!='SOURCE_EXTENSION_CLASSICAL_BANK_COMPLETE' or value.get('synthetic') is not False or value.get('arm')!=ctx.arm:
        raise DownstreamError('real_own_five_source_bank_receipt_required')
    resources,claim,budget=_resource_proof(ctx.root,value['resources'],ctx.arm)
    payload=_json(checked(ctx.root,value['payload']));identity=_json(checked(ctx.root,payload['identity']))
    if any(identity.get(k)!=v for k,v in {'arm':ctx.arm,'method':method,'configuration':configuration,
        'input_contract_sha256':ctx.signal_sha,'cache_sha256':ctx.source_cache_sha256,
        'UID_set_sha256':ctx.training_UID_set_sha256,'code':ctx.code,'source_dataset_ids':sorted(_datasets(ctx.arm))}.items()):
        raise DownstreamError('source_bank_arm_allowlist_configuration_or_code_changed')
    key=_bank_key(method,configuration)
    if resources['ledger']['bank_fits'].get(key)!= {'identity_sha256':digest(identity),'dispatched':True}:
        raise DownstreamError('source_bank_dispatch_proof_missing')
    if not _positive(payload.get('elapsed_seconds')) or payload['elapsed_seconds']>budget['caps']['source_bank_per_job']:
        raise DownstreamError('source_bank_elapsed_invalid_or_overrun')
    checked(ctx.root,identity['source_input']);cp=checked(ctx.root,payload['checkpoint']);checked(ctx.root,payload['sidecar']);saved=verified_checkpoint(cp)
    bank=_bank(method,{k:v.numpy() for k,v in saved['arrays'].items()},saved['metadata'])
    if bank.state_sha256!=saved['state_sha256'] or not bank.metadata.get('converged'):raise DownstreamError('source_bank_numeric_integrity_or_convergence_failed')
    return bank


def configuration(method,index):
    if method not in PRIMARY_METHODS or type(index) is not int or not 0<=index<len(registered_grid(method)):
        raise DownstreamError('registered_configuration_index_required')
    return dict(registered_grid(method)[index])


def _meta_key(job):return job.identity


def engineering_job(method):
    if method not in LEARNED:raise DownstreamError('registered_learned_engineering_family_required')
    row=max(registered_grid(method),key=lambda c:(c.get('dimension',c.get('rank')),c['radius']))
    return MetaJob(method,row.get('dimension',row.get('rank')),row['radius'],SEEDS[0],'engineering')


def _meta_folder(root,arm,job):
    return _owned(root,arm,f'runs/source_extension_downstream/{arm}/meta/{job.identity}')


def _read_training_state(root,item,identity):
    path=checked(root,item);side=path.with_suffix(path.suffix+'.sha256.json')
    value=_json(side)
    if value.get('sha256')!=file_digest(path) or value.get('bytes')!=path.stat().st_size or value.get('identity_sha256')!=digest(identity):
        raise DownstreamError('meta_checkpoint_sidecar_or_identity_changed')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('schema_version')!='paper3-trusted-training-state-v1' or saved.get('identity')!=identity or saved.get('identity_sha256')!=digest(identity):
        raise DownstreamError('own_meta_checkpoint_identity_required')
    try:
        rng=saved['rng'];random.Random().setstate(rng['python']);n=rng['numpy']
        np.random.RandomState().set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
        torch.Generator(device='cpu').set_state(rng['torch_cpu'])
        if (not isinstance(rng['cuda'],list) or not rng['cuda']
                or any(type(v) is not torch.Tensor or v.dtype!=torch.uint8 or v.ndim!=1 or not v.numel() for v in rng['cuda'])):
            raise ValueError('cuda_rng')
    except (ValueError,TypeError,KeyError,RuntimeError,IndexError):raise DownstreamError('actual_complete_resumable_meta_RNG_required') from None
    return saved


def _meta_context_identity(ctx,job,train,val,runtime,budget_ref,seconds):
    return {'schema_version':SCHEMA,'kind':'five_source_full_unroll','arm':ctx.arm,'job':asdict(job),
        'context':ctx.production_identity,'source_f0':ctx.source_receipt,'reference':ctx.reference_receipt,
        'input_contract_sha256':ctx.signal_sha,'split_sha256':ctx.split_sha,'source_UID_set_sha256':ctx.training_UID_set_sha256,
        'source_cache_sha256':ctx.source_cache_sha256,'meta_train':_catalog_binding(train),
        'meta_validation':_catalog_binding(val),'runtime':runtime,'budget':budget_ref,'max_wall_seconds':seconds,
        'full_unroll':True,'AMP':False,'internal_allowed':False,'external_allowed':False,'code':ctx.code}


def _meta_profile(ctx,catalog,method,runtime):
    item=catalog.get('meta_engineering',{}).get(method)
    if item is None:raise DownstreamError('actual_own_arm_same_method_engineering100_required')
    value=verify_meta(ctx.root,item,catalog)
    old=MetaJob(**value['job'])
    if old.phase!='engineering' or old.method!=method or value['runtime']!=runtime or not _positive(value.get('measured_wall_seconds')):
        raise DownstreamError('actual_matching_own_meta_engineering_profile_required')
    return value,item


def _meta_selection(root,catalog,method,*,execute=False):
    if method not in LEARNED:raise DownstreamError('learned_meta_selection_required')
    rows=[]
    for cfg in registered_grid(method):
        job=MetaJob(method,cfg.get('dimension',cfg.get('rank')),cfg['radius'])
        item=catalog.get('meta',{}).get(job.identity)
        if item is None:raise DownstreamError('all_nine_real_meta_screening_jobs_required')
        value=verify_meta(root,item,catalog)
        rows.append((value,item))
    # The same complete downstream grid selects both confirmation f0 pipelines.
    # Do not retune a learned family using a second, subtly different selector.
    configuration_ref=catalog.get('configurations',{}).get(method)
    if configuration_ref is None:raise DownstreamError('actual_complete_downstream_grid_configuration_required')
    configured=_verify_configuration(root,configuration_ref,catalog)
    cfg=configured['configuration']
    winner=next(pair for pair in rows if (pair[0]['job']['dimension'],pair[0]['job']['radius'])==(cfg.get('dimension',cfg.get('rank')),cfg['radius']))
    value={'schema_version':SCHEMA,'status':'SOURCE_EXTENSION_META_CONFIGURATION_FROZEN','arm':catalog['arm'],
        'method':method,'seed':SEEDS[0],'dimension':winner[0]['job']['dimension'],'radius':winner[0]['job']['radius'],
        'screening_receipts':[item for _,item in rows],'selected_screening_receipt':winner[1],
        'downstream_configuration':configuration_ref,
        'tie_order':'participant_BA_desc/BCE_asc/checkpoint_asc/dimension_asc/radius_asc',
        'internal_allowed':False,'external_allowed':False}
    if execute:
        path=_owned(root,catalog['arm'],f"runs/source_extension_downstream/{catalog['arm']}/meta_selections/{method}.json")
        immutable_json(path,value);item=descriptor(root,path);_update_catalog(root,catalog,'meta_selections',method,item);return item
    return value


@threadpool_limits.wrap(limits=1)
def execute_meta(root,catalog,job,*,execute=False,user_started_gpu=False):
    if type(job) is not MetaJob or job.seed not in DEVELOPMENT_SEEDS:raise DownstreamError('registered_three_seed_meta_job_required')
    if job.phase=='engineering' and job!=engineering_job(job.method):raise DownstreamError('unique_largest_registered_engineering_job_required')
    if not execute:return {**build_plan(_catalog(root,catalog)['arm']),'stage':'meta','job':asdict(job),'status':'PLANNED_NOT_EXECUTED'}
    if not user_started_gpu:raise PermissionError('execute_and_later_user_started_GPU_both_required')
    _require_owned_stage()
    root=Path(root).resolve(strict=True);catalog=_latest_catalog(root,catalog);ctx,train,val=load_context(root,catalog,job.seed)
    runtime=_runtime();_gpu(ctx);_production(ctx)
    if job.phase=='confirmation':
        item=catalog.get('meta_selections',{}).get(job.method)
        selected=_meta_selection(root,catalog,job.method)
        if item is None or _json(checked(root,item))!=selected or (job.dimension,job.radius)!=(selected['dimension'],selected['radius']):
            raise DownstreamError('confirmation_must_use_exact_frozen_meta_configuration')
    run=ArmRun(root,catalog,runtime);folder=_meta_folder(root,ctx.arm,job);folder.mkdir(parents=True,exist_ok=True)
    complete_ref=None
    try:
        proof=None
        if job.phase=='engineering':seconds=run.budget['caps']['meta_engineering'][job.method]
        else:
            proof,profile_ref=_meta_profile(ctx,catalog,job.method,runtime)
            seconds=min(run.budget['caps']['meta_science_per_job'][job.method],SAFETY_FACTOR*proof['measured_wall_seconds']*10+120)
        identity=_meta_context_identity(ctx,job,train,val,runtime,run.budget_ref,seconds)
        if proof is not None:identity['measured_own_engineering']=profile_ref
        if job.phase=='confirmation':identity['frozen_meta_selection']=item
        path=folder/'receipt.json'
        if path.exists():
            ref=descriptor(root,path);verify_meta(root,ref,catalog)
            _update_catalog(root,catalog,'meta_engineering' if job.phase=='engineering' else 'meta',job.method if job.phase=='engineering' else job.identity,ref)
            return {'status':'ALREADY_COMPLETE_VERIFIED','receipt':ref,'new_fits':0}
        prior=run.state['stages'].get(job.identity)
        if prior is not None and prior.get('identity_sha256')!=digest(identity):raise DownstreamError('meta_job_dispatch_identity_changed')
        if prior is not None:
            for name in ('identity.json','latest_checkpoint.json','progress.json','wall_budget.json'):
                if not (folder/name).is_file():raise DownstreamError('dispatched_meta_recovery_identity_checkpoint_or_ledger_missing')
        elif any((folder/name).exists() for name in ('identity.json','latest_checkpoint.json','progress.json','wall_budget.json')):
            raise DownstreamError('meta_work_exists_without_arm_dispatch')
        run.state['stages'][job.identity]={'identity_sha256':digest(identity),'kind':'meta','dispatched':True};run.save()
        immutable_json(folder/'identity.json',identity)
        random.seed(job.seed);np.random.seed(job.seed);torch.manual_seed(job.seed);torch.cuda.manual_seed_all(job.seed)
        if job.method in {'B10','B11'}:
            initial=lowrank.initialize_lowrank(job.dimension,job.seed,device='cuda:0').requires_grad_(True)
            config=lowrank.LowRankConfig(rank=job.dimension,radius=job.radius,pct_weight=.1 if job.method=='B11' else 0.)
            kernel=lambda p,episodes:lowrank.lowrank_outer_step(ctx.source,ctx.geometry,p,episodes,config=config,method=job.method)
        else:
            initial=spatial.initialize_dictionary(job.dimension,job.seed,device='cuda:0').requires_grad_(True)
            config=InnerConfig(radius=job.radius,pct_weight=.1 if job.method=='P+PCT' else 0.)
            kernel=lambda p,episodes:outer_step(ctx.source,ctx.geometry,p,episodes,config=config,family='additive' if job.method=='B8' else 'exp')
        sampler=DeterministicEpisodeSampler(train,job.seed,device='cuda:0')
        resume=None
        if (folder/'latest_checkpoint.json').exists():
            latest=_json(folder/'latest_checkpoint.json')
            if latest.get('relative_to_job_output') is not True or Path(latest.get('path','')).name!=latest.get('path'):
                raise DownstreamError('same_job_meta_checkpoint_required')
            item={'path':(folder/latest['path']).relative_to(root).as_posix(),'sha256':latest['sha256']}
            _read_training_state(root,item,identity);resume=checked(root,item)
        def measured_step(parameter,episodes):
            _production(ctx);run.guard()
            calls=run.state['meta_calls'].get(job.identity,0)+1
            if calls>2*job.updates:raise DownstreamError('same_job_physical_meta_call_limit_exhausted')
            run.observe('meta_calls',job.identity,calls)
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            torch.cuda.reset_peak_memory_stats(0);start=time.monotonic();begin.record()
            result,info=kernel(parameter,episodes);end.record();end.synchronize()
            actual=sum(10 for e in episodes if bool((e.support.y==0).any() and (e.support.y==1).any()))
            run.guard()
            return result,{**info,'inner_updates_actual':actual,'gpu_stream_elapsed_ms':begin.elapsed_time(end),
                'end_to_end_wall_seconds':time.monotonic()-start,'max_memory_allocated_bytes':torch.cuda.max_memory_allocated(0),
                'max_memory_reserved_bytes':torch.cuda.max_memory_reserved(0)}
        stage_start=time.monotonic()
        with _deadline(run.wall.remaining_seconds),torch.autocast(device_type='cuda',enabled=False):
            _,result=run_outer_loop(parameter=initial,sampler=sampler,step=measured_step,
                evaluate=lambda p:evaluate_development(ctx.source,ctx.geometry,p,val,job),identity=identity,output=folder,
                total_updates=job.updates,max_wall_seconds=seconds,resume=resume,cuda=True)
        _production(ctx);run.guard()
        if result['status']!='UPDATE_BUDGET_COMPLETED':return {'status':'PAUSED_NOT_COMPLETE','job':asdict(job),'result':result}
        selected=result['selected_checkpoint'];selected_path=folder/f"outer-{selected['outer_update']:04d}.pt"
        cp={'path':selected_path.relative_to(root).as_posix(),'sha256':file_digest(selected_path)}
        _read_training_state(root,cp,identity)
        value={'schema_version':SCHEMA,'status':'SOURCE_EXTENSION_META_ENGINEERING_COMPLETE' if job.phase=='engineering' else 'SOURCE_EXTENSION_META_COMPLETE',
            'synthetic':False,'arm':ctx.arm,'job':asdict(job),'identity':descriptor(root,folder/'identity.json'),
            'source_f0':ctx.source_receipt,'reference':ctx.reference_receipt,'runtime':runtime,'context':ctx.production_identity,
            'full_unroll':True,'validation':result['validation'],'selected_checkpoint':selected,
            'selected_parameter':cp,'final_checkpoint':descriptor(root,folder/f'outer-{job.updates:04d}.pt'),
            'progress':descriptor(root,folder/'progress.json'),'job_closed_wall':descriptor(root,folder/'wall_budget.json'),
            'physical_outer_calls':run.state['meta_calls'][job.identity],
            'measured_wall_seconds':_json(folder/'wall_budget.json')['durable_wall_seconds'],
            'scientific_fit_completed':job.phase!='engineering','internal_allowed':False,'external_allowed':False}
        complete_ref=(path,value)
    finally:run.close()
    path,value=complete_ref;value['resources']=run.snapshot(folder);immutable_json(path,value)
    item=descriptor(root,path);verify_meta(root,item,catalog)
    _update_catalog(root,catalog,'meta_engineering' if job.phase=='engineering' else 'meta',job.method if job.phase=='engineering' else job.identity,item)
    return {'status':value['status'],'receipt':item,'new_scientific_meta_fits':int(job.phase!='engineering')}


def verify_meta(root,item,catalog):
    root=Path(root).resolve(strict=True);catalog=_catalog(root,catalog);value=_json(checked(root,item));job=MetaJob(**value['job'])
    if job.phase=='engineering' and job!=engineering_job(job.method):raise DownstreamError('canonical_own_engineering_profile_required')
    ctx,train,val=load_context(root,catalog,job.seed);_production(ctx)
    resources,claim,budget=_resource_proof(root,value['resources'],ctx.arm)
    identity=_json(checked(root,value['identity']));seconds=identity.get('max_wall_seconds')
    expected=_meta_context_identity(ctx,job,train,val,value['runtime'],value['resources'] and resources['budget'],seconds)
    if job.phase!='engineering':
        profile,profile_item=_meta_profile(ctx,catalog,job.method,value['runtime'])
        expected['measured_own_engineering']=profile_item
        maximum=min(budget['caps']['meta_science_per_job'][job.method],SAFETY_FACTOR*profile['measured_wall_seconds']*10+120)
    else:maximum=budget['caps']['meta_engineering'][job.method]
    if job.phase=='confirmation':
        selection=catalog.get('meta_selections',{}).get(job.method)
        selected=_meta_selection(root,catalog,job.method)
        if selection is None or _json(checked(root,selection))!=selected or (job.dimension,job.radius)!=(selected['dimension'],selected['radius']):
            raise DownstreamError('confirmation_exact_frozen_meta_selection_required')
        expected['frozen_meta_selection']=selection
    if identity!=expected or seconds!=maximum:raise DownstreamError('own_meta_identity_budget_source_or_catalog_changed')
    expected_status='SOURCE_EXTENSION_META_ENGINEERING_COMPLETE' if job.phase=='engineering' else 'SOURCE_EXTENSION_META_COMPLETE'
    if (value.get('schema_version')!=SCHEMA or value.get('status')!=expected_status or value.get('synthetic') is not False
            or value.get('arm')!=ctx.arm or value.get('context')!=ctx.production_identity or value.get('full_unroll') is not True
            or value.get('internal_allowed') is not False or value.get('external_allowed') is not False
            or value.get('scientific_fit_completed')!=(job.phase!='engineering')
            or value.get('source_f0')!=ctx.source_receipt or value.get('reference')!=ctx.reference_receipt):
        raise DownstreamError('actual_own_source_extension_meta_receipt_required')
    wall=_json(checked(root,value['job_closed_wall']));_closed_wall(wall,digest(identity),maximum)
    if value.get('measured_wall_seconds')!=wall['durable_wall_seconds'] or not _positive(value['measured_wall_seconds']):
        raise DownstreamError('actual_measured_meta_wall_required')
    history=value.get('validation',[])
    if [r.get('outer_update') for r in history]!=list(range(50,job.updates+1,50)) or choose_checkpoint(history)!=value.get('selected_checkpoint'):
        raise DownstreamError('all_registered_validation_checkpoints_required')
    for row in history:
        summary=participant_equal_metrics(row['sessions'],expected_participants=val.partition_people)
        if any(row.get(k)!=v for k,v in summary.items()):raise DownstreamError('meta_validation_participant_denominator_or_metrics_changed')
    selected=value['selected_checkpoint']['outer_update'];final=_read_training_state(root,value['final_checkpoint'],identity)
    chosen=_read_training_state(root,value['selected_parameter'],identity)
    for saved,cursor in ((final,job.updates),(chosen,selected)):
        state=saved['state'];parameter=state['parameter'];dim=10 if job.method in {'B10','B11'} else 99
        if state.get('outer_cursor')!=cursor or parameter.dtype!=torch.float64 or parameter.shape!=(dim,job.dimension) or not torch.isfinite(parameter).all():
            raise DownstreamError('actual_meta_checkpoint_cursor_or_parameter_invalid')
        if not torch.allclose(parameter.T@parameter,torch.eye(job.dimension,dtype=torch.float64),atol=1e-10,rtol=1e-10):
            raise DownstreamError('learned_parameter_not_orthonormal')
        sampler=DeterministicEpisodeSampler(train,job.seed,device='cpu');sampler.load_state_dict(state['sampler'])
        if sampler.cursor!=cursor*4:raise DownstreamError('exact_meta_episode_sampler_checkpoint_cursor_required')
    state=final['state'];updates=state.get('update_receipts',[])
    if len(updates)!=job.updates or [r.get('outer_update') for r in updates]!=list(range(1,job.updates+1)) or state.get('validation')!=history:
        raise DownstreamError('full_actual_meta_update_history_required')
    for row in updates:
        if row.get('full_unroll') is not True or type(row.get('inner_updates_actual')) is not int or not 0<=row['inner_updates_actual']<=40 or not _positive(row.get('gpu_stream_elapsed_ms')) or not _positive(row.get('end_to_end_wall_seconds')):
            raise DownstreamError('real_full_unroll_measured_update_evidence_required')
    physical=resources['ledger'].get('meta_calls',{}).get(job.identity)
    if type(physical) is not int or not job.updates<=physical<=2*job.updates or physical!=value.get('physical_outer_calls'):
        raise DownstreamError('meta_physical_attempt_ledger_changed')
    progress=_json(checked(root,value['progress']))
    if (progress.get('status')!='UPDATE_BUDGET_COMPLETED' or progress.get('outer_updates_completed')!=job.updates
            or progress.get('identity_sha256')!=digest(identity) or progress.get('validation_checkpoints')!=job.updates//50):
        raise DownstreamError('actual_completed_same_job_meta_progress_required')
    return value


def load_learned(ctx,catalog,method,config):
    _production(ctx);job=MetaJob(method,config.get('dimension',config.get('rank')),config['radius'],ctx.seed,'screening' if ctx.seed==SEEDS[0] else 'confirmation')
    item=catalog.get('meta',{}).get(job.identity)
    if item is None:raise DownstreamError('actual_same_arm_seed_configuration_learned_artifact_required')
    value=verify_meta(ctx.root,item,catalog);saved=_read_training_state(ctx.root,value['selected_parameter'],_json(checked(ctx.root,value['identity'])))
    parameter=saved['state']['parameter'].detach().to(next(ctx.source.parameters()).device)
    return parameter,item,value['selected_checkpoint']['outer_update']


def _validation_job(method,index,seed,phase):
    config=configuration(method,index)
    if seed not in DEVELOPMENT_SEEDS or phase not in {'screening','confirmation'} or (phase=='screening')!=(seed==SEEDS[0]):
        raise DownstreamError('registered_validation_phase_and_seed_required')
    value={'method':method,'configuration_index':index,'configuration':config,'seed':seed,'phase':phase}
    return {**value,'identity_sha256':digest(value)}


def _dependency(ctx,catalog,method,config):
    parameter=bank=None;learned_ref=bank_ref=None;outer=0
    if method in LEARNED:parameter,learned_ref,outer=load_learned(ctx,catalog,method,config)
    elif method in {'B4','B5'}:
        cfg={'m':config['m']} if method=='B4' else {'shrinkage':config['shrinkage']}
        bank_ref=catalog.get('source_banks',{}).get(_bank_key(method,cfg))
        if bank_ref is None:raise DownstreamError('own_arm_classical_source_bank_required')
        bank=load_source_bank(ctx,bank_ref,method,cfg)
    return parameter,bank,{'learned':learned_ref,'source_bank':bank_ref,'selected_meta_outer_update':outer}


def _support_binding(support):
    support.__post_init__()
    if support.role!='meta_validation_support' or len(support.acquired_event_uids)!=20:
        raise DownstreamError('only_true_twenty_acquired_validation_prefix_allowed')
    return {'valid_support_event_uids':list(support.event_uids),'all_twenty_acquired_event_uids':list(support.acquired_event_uids),
        'support_x_sha256':tensor_sha256(support.x),'support_y_sha256':tensor_sha256(support.y),
        'signal_sha256':support.protocol_freeze_sha,'role':support.role}


def _candidate(ctx,job,plan,support,dependencies,budget_ref,purpose):
    if purpose not in {'calibration_profile','validation'} or plan.role!='meta_validation':
        raise PermissionError('development_only_candidate_required')
    return {'schema_version':SCHEMA,'kind':purpose,'arm':ctx.arm,'context':ctx.production_identity,
        'job':job,'plan':plan.to_dict(),'support':_support_binding(support),'dependencies':dependencies,
        'budget':budget_ref,'code':ctx.code,'internal_allowed':False,'external_allowed':False}


def _issue_fitted(ctx,state,candidate):
    _production(ctx)
    if state.context_sha256!=ctx.identity or state.support_event_uids!=tuple(candidate['support']['valid_support_event_uids']) or state.acquired_event_uids!=tuple(candidate['support']['all_twenty_acquired_event_uids']):
        raise DownstreamError('actual_fitted_support_identity_mismatch')
    key=id(state)
    _FIT_SEALS[key]=(weakref.ref(state,lambda _ref:_FIT_SEALS.pop(key,None)),digest(candidate),ctx.production_identity,state.state_sha256)
    return state


def _target_once(ctx,catalog,run,job,plan,folder,seconds,purpose):
    """One durable numerical dispatch; uncertain target fits never run again."""
    _production(ctx);method=job['method'];config=job['configuration']
    if not _positive(seconds):raise DownstreamError('positive_measured_candidate_limit_required')
    support_catalog=load_episode_catalog(ctx.root,role='meta_validation',purpose='model_selection')
    actual=support_catalog.plan(plan.participant_id,plan.session_id)
    if actual.to_dict()!=plan.to_dict():raise DownstreamError('candidate_plan_changed')
    support=support_catalog.materialize_support(actual,device='cpu')
    parameter,bank,dependencies=_dependency(ctx,catalog,method,config)
    candidate=_candidate(ctx,job,actual,support,dependencies,run.budget_ref,purpose)
    folder.mkdir(parents=True,exist_ok=True);key=digest(candidate);committed=(folder/'fit_receipt.json').exists()
    dispatched=run.dispatch_once('target_fits',key,candidate,committed=committed)
    immutable_json(folder/'candidate.json',candidate)
    if not dispatched:
        fitted=load_adapter(ctx,descriptor(ctx.root,folder/'fit_receipt.json'),candidate)
        timing=_json(folder/'timing.json')
        if timing.get('candidate_sha256')!=key or not _positive(timing.get('calibration_wall_seconds')) or timing['calibration_wall_seconds']>seconds:
            raise DownstreamError('committed_target_measured_timing_missing_or_overrun')
        return fitted,descriptor(ctx.root,folder/'fit_receipt.json'),timing,candidate
    # Fixed, label-free forward warmup does not perform an optimizer update.
    with torch.no_grad(),torch.autocast(device_type='cuda',enabled=False):
        warm=support.x[:min(2,len(support.x))].to('cuda:0')
        if len(warm):ctx.source.source_logits(warm)
    torch.cuda.synchronize();run.guard();limit=min(seconds,run.wall.remaining_seconds)
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    started=time.monotonic();begin.record();torch.cuda.reset_peak_memory_stats(0)
    with _deadline(limit),torch.autocast(device_type='cuda',enabled=False):
        # Actual timing includes support transfer and method artifact checks.
        support=support_catalog.materialize_support(actual,device='cpu' if method in {'B4','B5'} else 'cuda:0')
        parameter,bank,actual_dependencies=_dependency(ctx,catalog,method,config)
        if actual_dependencies!=dependencies:raise DownstreamError('candidate_dependency_changed_during_dispatch')
        fitted=fit_support_math(ctx,support,method,config,learned=parameter,bank=bank)
        end.record();end.synchronize()
    elapsed=time.monotonic()-started
    if not _positive(elapsed) or elapsed>limit:raise DownstreamError('measured_target_fit_overrun_no_completion')
    _production(ctx);run.guard()
    timing={'schema_version':SCHEMA,'candidate_sha256':key,'calibration_wall_seconds':elapsed,
        'gpu_stream_elapsed_seconds':begin.elapsed_time(end)/1000,'method_compute_device':'cpu' if method in {'B4','B5'} else 'cuda:0',
        'actual_optimizer_updates':fitted.updates,'fallback_reason':fitted.fallback_reason,
        'trainable_parameters':fitted.trainable_parameters,'max_memory_allocated_bytes':torch.cuda.max_memory_allocated(0),
        'max_memory_reserved_bytes':torch.cuda.max_memory_reserved(0),'measured':True,'runtime':run.runtime,
        'acquired_label_cost':20,'valid_support_trials':len(support.event_uids),'new_fit_attempts':1,
        'source_state_unchanged':True,'internal_allowed':False,'external_allowed':False}
    immutable_json(folder/'timing.json',timing)
    _issue_fitted(ctx,fitted,candidate);fit_ref=save_adapter(ctx,fitted,folder,candidate)
    return fitted,fit_ref,timing,candidate


def _verify_target(ctx,item,candidate,resources):
    state=load_adapter(ctx,item,candidate);key=digest(candidate)
    expected={'identity_sha256':key,'dispatched':True}
    if resources['ledger'].get('target_fits',{}).get(key)!=expected:
        raise DownstreamError('target_fit_actual_dispatch_ledger_required')
    folder=checked(ctx.root,item).parent;timing=_json(folder/'timing.json')
    if timing.get('candidate_sha256')!=key or timing.get('measured') is not True or not _positive(timing.get('calibration_wall_seconds')) or timing.get('actual_optimizer_updates')!=state.updates:
        raise DownstreamError('target_real_measurement_and_fit_binding_invalid')
    return state,timing


def _profile_session(catalog):
    """First audited two-class B20 support; never materialize a query."""
    for plan in catalog.plans:
        if (plan.query_eligible and len(plan.acquired_event_uids)==20
                and len(plan.support_event_uids)>=2 and plan.fallback_reason is None):
            support=catalog.materialize_support(plan,device='cpu');support.__post_init__()
            if (support.role!='meta_validation_support' or support.event_uids!=plan.support_event_uids
                    or support.acquired_event_uids!=plan.acquired_event_uids
                    or not all(bool((support.y==label).any()) for label in (0,1))):
                raise DownstreamError('profile_metadata_and_actual_two_class_B20_support_disagree')
            return plan,support
    raise DownstreamError('fixed_validation_catalog_has_no_two_class_B20_profile_session')


def _profile_fit_required(state,timing):
    if state.fallback_reason is not None or timing.get('fallback_reason','MISSING') is not None:
        raise DownstreamError('fallback_cannot_supply_real_calibration_profile')


@threadpool_limits.wrap(limits=1)
def execute_profile(root,catalog,*,execute=False,user_started_gpu=False):
    if not execute:return {**build_plan(_catalog(root,catalog)['arm']),'stage':'profile','status':'PLANNED_NOT_EXECUTED'}
    if not user_started_gpu:raise PermissionError('execute_and_later_user_started_GPU_both_required')
    _require_owned_stage()
    root=Path(root).resolve(strict=True);catalog=_latest_catalog(root,catalog);ctx,train,val=load_context(root,catalog)
    runtime=_runtime();_gpu(ctx);_production(ctx)
    folder=_owned(root,ctx.arm,f'runs/source_extension_downstream/{ctx.arm}/calibration_profile')
    folder.mkdir(parents=True,exist_ok=True);path=folder/'receipt.json'
    if path.exists():
        ref=descriptor(root,path);verify_profile(root,ref,catalog);_update_catalog(root,catalog,'profiles','all118',ref)
        return {'status':'ALREADY_COMPLETE_VERIFIED','receipt':ref,'new_fits':0}
    plan,_=_profile_session(val);run=ArmRun(root,catalog,runtime);records=[]
    try:
        wall_identity={'context':ctx.production_identity,'runtime':runtime,'budget':run.budget_ref,'profile_session':plan.to_dict()}
        with _stage_wall(run,folder,'calibration-profile',wall_identity,run.budget['caps']['calibration_profile']) as stage_wall:
            # Bootstrap bounds originate from actual primary same-device profiles;
            # every subsequent validation limit uses this arm's actual timings.
            per=min(run.budget['caps']['calibration_profile']/118,run.wall.remaining_seconds)
            for method in PRIMARY_METHODS:
                for index in range(len(registered_grid(method))):
                    job=_validation_job(method,index,SEEDS[0],'screening');dest=folder/job['identity_sha256']
                    state,item,timing,candidate=_target_once(ctx,catalog,run,job,plan,dest,per,'calibration_profile')
                    _profile_fit_required(state,timing)
                    records.append({'job':job,'candidate':descriptor(root,dest/'candidate.json'),'fitted':item,
                        'timing':descriptor(root,dest/'timing.json'),'calibration_wall_seconds':timing['calibration_wall_seconds']})
                    run.guard()
                    if stage_wall.exhausted:raise DownstreamError('profile_cumulative_wall_exhausted')
    finally:run.close()
    value={'schema_version':SCHEMA,'status':'SOURCE_EXTENSION_CALIBRATION_PROFILE_COMPLETE','synthetic':False,
        'arm':ctx.arm,'context':ctx.production_identity,'runtime':runtime,'profile_session':plan.to_dict(),
        'catalog':_catalog_binding(val),'records':records,'source_feature_dim':ctx.source.feature_dim,
        'geometry_sha256':ctx.geometry.sha256(),'training_UID_set_sha256':ctx.training_UID_set_sha256,
        'resources':run.snapshot(folder),'stage_wall':descriptor(root,folder/'stage_wall.json'),
        'internal_allowed':False,'external_allowed':False}
    immutable_json(path,value);item=descriptor(root,path);verify_profile(root,item,catalog)
    _update_catalog(root,catalog,'profiles','all118',item)
    return {'status':value['status'],'receipt':item,'actual_profile_fits':118}


def verify_profile(root,item,catalog):
    root=Path(root).resolve(strict=True);catalog=_catalog(root,catalog);value=_json(checked(root,item));ctx,train,val=load_context(root,catalog)
    _production(ctx);resources,claim,budget=_resource_proof(root,value['resources'],ctx.arm)
    expected={'schema_version':SCHEMA,'status':'SOURCE_EXTENSION_CALIBRATION_PROFILE_COMPLETE','synthetic':False,
        'arm':ctx.arm,'context':ctx.production_identity,'catalog':_catalog_binding(val),
        'source_feature_dim':ctx.source.feature_dim,'geometry_sha256':ctx.geometry.sha256(),
        'training_UID_set_sha256':ctx.training_UID_set_sha256,'internal_allowed':False,'external_allowed':False}
    if any(value.get(k)!=v for k,v in expected.items()) or value.get('runtime')!=claim.get('runtime'):
        raise DownstreamError('actual_own_all_method_profile_required')
    plan,support=_profile_session(val)
    if value.get('profile_session')!=plan.to_dict():raise DownstreamError('fixed_profile_session_changed')
    wall_identity={'context':ctx.production_identity,'runtime':value['runtime'],'budget':resources['budget'],'profile_session':plan.to_dict()}
    _verify_stage_wall(root,value['stage_wall'],wall_identity,budget['caps']['calibration_profile'],resources,'calibration-profile')
    expected_jobs={(m,i) for m in PRIMARY_METHODS for i in range(len(registered_grid(m)))};seen=set();maxima={}
    for record in value.get('records',[]):
        job=record['job'];key=(job['method'],job['configuration_index'])
        if key not in expected_jobs or key in seen or job!=_validation_job(*key,SEEDS[0],'screening'):
            raise DownstreamError('all_unique_registered118_profile_settings_required')
        candidate=_json(checked(root,record['candidate']))
        # Signal tensor hash is portable across devices; producer used the same
        # catalog bytes and integer support labels, never query observations.
        deps=_dependency(ctx,catalog,job['method'],job['configuration'])[2]
        expected_candidate=_candidate(ctx,job,plan,support,deps,resources['budget'],'calibration_profile')
        if candidate!=expected_candidate:raise DownstreamError('profile_candidate_source_support_or_config_changed')
        state,timing=_verify_target(ctx,record['fitted'],candidate,resources)
        _profile_fit_required(state,timing)
        checked(root,record['timing'])
        if record['calibration_wall_seconds']!=timing['calibration_wall_seconds'] or timing['runtime']!=value['runtime'] or timing['calibration_wall_seconds']>budget['caps']['calibration_profile']/118:
            raise DownstreamError('profile_actual_timing_or_runtime_changed')
        maxima[job['method']]=max(maxima.get(job['method'],0),timing['calibration_wall_seconds']);seen.add(key)
    if seen!=expected_jobs:raise DownstreamError('all118_profile_settings_required')
    return {'receipt':descriptor(root,checked(root,item)),'runtime':value['runtime'],'maximum_calibration_seconds':maxima,
        'geometry_sha256':ctx.geometry.sha256(),'source_feature_dim':ctx.source.feature_dim,'training_UID_set_sha256':ctx.training_UID_set_sha256}


def _production_predict(ctx,job,plan,fit_ref,candidate):
    """Only the actual issued catalog and committed fit can issue a prediction."""
    _production(ctx);state=load_adapter(ctx,fit_ref,candidate)
    catalog=load_episode_catalog(ctx.root,role='meta_validation',purpose='model_selection')
    actual=catalog.plan(plan.participant_id,plan.session_id)
    if actual.to_dict()!=plan.to_dict():raise DownstreamError('prediction_typed_catalog_plan_changed')
    inputs=catalog.materialize(actual,device='cpu' if job['method'] in {'B4','B5'} else 'cuda:0').prediction_inputs
    result=predict_math(ctx,state,inputs)
    numeric=_tensor_tree_sha({'logits':result.logits,'probabilities':result.probabilities,'labels':result.labels})
    key=id(result)
    seal={'context':ctx.production_identity,'job':copy.deepcopy(job),'plan':plan.to_dict(),'fitted':copy.deepcopy(fit_ref),
        'candidate_sha256':digest(candidate),'query_x_sha256':tensor_sha256(inputs.x),'numeric_sha256':numeric}
    _PREDICTION_SEALS[key]=(weakref.ref(result,lambda _ref:_PREDICTION_SEALS.pop(key,None)),seal)
    return result


def _commit_prediction(ctx,job,plan,prediction,fit_ref,candidate,folder):
    _production(ctx)
    if type(prediction) is not PredictionResult:raise DownstreamError('typed_prediction_result_required')
    issued=_PREDICTION_SEALS.get(id(prediction))
    expected={'context':ctx.production_identity,'job':job,'plan':plan.to_dict(),'fitted':fit_ref,'candidate_sha256':digest(candidate)}
    if (issued is None or issued[0]() is not prediction or any(issued[1].get(k)!=v for k,v in expected.items())
            or issued[1]['numeric_sha256']!=_tensor_tree_sha({'logits':prediction.logits,'probabilities':prediction.probabilities,'labels':prediction.labels})):
        raise DownstreamError('actual_production_prediction_dispatch_required')
    try:relative=Path(folder).relative_to(ctx.root).as_posix()
    except ValueError:raise DownstreamError('owned_prediction_output_required') from None
    folder=_owned(ctx.root,ctx.arm,relative)
    if tuple(prediction.event_uids)!=tuple(plan.query_event_uids) or prediction.protocol_freeze_sha!=INPUT_SHA:
        raise DownstreamError('exact_full_future_query_required')
    shape=(len(plan.query_event_uids),2)
    logits=prediction.logits.detach().cpu().numpy();proba=prediction.probabilities.detach().cpu().numpy();labels=prediction.labels.detach().cpu().numpy()
    if logits.shape!=shape or proba.shape!=shape or labels.shape!=(shape[0],) or not np.isfinite(logits).all() or not np.isfinite(proba).all():
        raise DownstreamError('full_finite_prediction_arrays_required')
    if not np.allclose(softmax(logits,axis=1),proba,atol=1e-6,rtol=1e-6) or not np.array_equal(np.argmax(logits,1),labels):
        raise DownstreamError('same_logits_prediction_binding_invalid')
    arrays={}
    for name,array in [('logits',logits),('probabilities',proba),('prediction',labels)]:
        path=folder/(name+'.npy');_npy_new(path,array);arrays[name]=descriptor(ctx.root,path)
    value={'schema_version':SCHEMA,'status':'SOURCE_EXTENSION_PREDICTIONS_COMMITTED','arm':ctx.arm,
        'job':job,'context':ctx.production_identity,'plan':plan.to_dict(),'candidate_sha256':digest(candidate),
        'fitted':fit_ref,'query_event_uids':list(plan.query_event_uids),'arrays':arrays,
        'query_x_sha256':issued[1]['query_x_sha256'],'numeric_sha256':issued[1]['numeric_sha256'],
        'query_labels_read_before_commit':False,'internal_allowed':False,'external_allowed':False}
    path=folder/'receipt.json';immutable_json(path,value);return descriptor(ctx.root,path)


def _prediction_arrays(ctx,item,job,plan,candidate,fit_ref):
    value=_json(checked(ctx.root,item))
    expected={'schema_version':SCHEMA,'status':'SOURCE_EXTENSION_PREDICTIONS_COMMITTED','arm':ctx.arm,
        'job':job,'context':ctx.production_identity,'plan':plan.to_dict(),'candidate_sha256':digest(candidate),
        'fitted':fit_ref,'query_event_uids':list(plan.query_event_uids),'query_labels_read_before_commit':False,
        'internal_allowed':False,'external_allowed':False}
    if any(value.get(k)!=v for k,v in expected.items()):raise DownstreamError('committed_prediction_identity_changed')
    data={k:np.load(checked(ctx.root,d),allow_pickle=False) for k,d in value['arrays'].items()}
    if set(data)!={'logits','probabilities','prediction'} or data['logits'].shape!=(len(plan.query_event_uids),2) or data['probabilities'].shape!=data['logits'].shape or data['prediction'].shape!=(len(plan.query_event_uids),):
        raise DownstreamError('full_committed_query_prediction_shape_required')
    if not np.isfinite(data['logits']).all() or not np.allclose(softmax(data['logits'],axis=1),data['probabilities'],atol=1e-6,rtol=1e-6) or not np.array_equal(data['logits'].argmax(1),data['prediction']):
        raise DownstreamError('committed_same_logits_prediction_invalid')
    catalog=load_episode_catalog(ctx.root,role='meta_validation',purpose='model_selection')
    inputs=catalog.materialize(plan,device='cpu').prediction_inputs
    numeric=_tensor_tree_sha({'logits':torch.from_numpy(data['logits']),'probabilities':torch.from_numpy(data['probabilities']),'labels':torch.from_numpy(data['prediction'])})
    if value.get('query_x_sha256')!=tensor_sha256(inputs.x) or value.get('numeric_sha256')!=numeric:
        raise DownstreamError('production_query_or_prediction_numerical_SHA_changed')
    return data


def _scores(episode,arrays):
    labels=episode.evaluation_labels.labels(purpose='development_evaluation').detach().cpu()
    logits=torch.from_numpy(np.asarray(arrays['logits']).copy());pred=logits.argmax(1)
    if len(labels)!=len(logits) or set(labels.tolist())!={0,1}:raise DownstreamError('actual_complete_LR_query_labels_required')
    return {'participant_id':episode.plan.participant_id,'session_id':episode.plan.session_id,
        'ba':sum(float((pred[labels==c]==c).double().mean()) for c in (0,1))/2,
        'balanced_bce':float(balanced_bce(logits,labels)),'query_trials':len(labels)}


@threadpool_limits.wrap(limits=1)
def execute_validation(root,catalog,method,index,seed,phase='screening',*,execute=False,user_started_gpu=False):
    job=_validation_job(method,index,seed,phase)
    if not execute:return {**build_plan(_catalog(root,catalog)['arm']),'stage':'evaluate','job':job,'status':'PLANNED_NOT_EXECUTED'}
    if not user_started_gpu:raise PermissionError('execute_and_later_user_started_GPU_both_required')
    _require_owned_stage()
    root=Path(root).resolve(strict=True);catalog=_latest_catalog(root,catalog);ctx,train,val=load_context(root,catalog,seed)
    runtime=_runtime();_gpu(ctx);_production(ctx)
    profile_item=catalog.get('profiles',{}).get('all118')
    if profile_item is None:raise DownstreamError('actual_own118_calibration_profile_required')
    profile=verify_profile(root,profile_item,catalog)
    if profile['runtime']!=runtime or profile['geometry_sha256']!=ctx.geometry.sha256() or profile['source_feature_dim']!=ctx.source.feature_dim or profile['training_UID_set_sha256']!=ctx.training_UID_set_sha256:
        raise DownstreamError('same_device_input_architecture_own_profile_required')
    selection_ref=None
    if phase=='confirmation':
        selection_ref=catalog.get('configurations',{}).get(method)
        if selection_ref is None:raise DownstreamError('immutable_complete_grid_configuration_required')
        selected=_verify_configuration(root,selection_ref,catalog)
        if selected['configuration_index']!=index:raise DownstreamError('confirmation_cannot_retune_configuration')
    folder=_owned(root,ctx.arm,f"runs/source_extension_downstream/{ctx.arm}/validation/{job['identity_sha256']}")
    folder.mkdir(parents=True,exist_ok=True);path=folder/'receipt.json'
    if path.exists():
        ref=descriptor(root,path);verify_validation(root,ref,catalog);_update_catalog(root,catalog,'validation',job['identity_sha256'],ref)
        return {'status':'ALREADY_COMPLETE_VERIFIED','receipt':ref,'new_fits':0}
    run=ArmRun(root,catalog,runtime);rows=[];records=[]
    try:
        per=SAFETY_FACTOR*profile['maximum_calibration_seconds'][method]+5
        max_job=min(run.budget['caps']['validation_per_job'],sum(p.query_eligible for p in val.plans)*(per+5)+120)
        wall_identity={'job':job,'context':ctx.production_identity,'profile':profile_item,'budget':run.budget_ref,'configuration_receipt':selection_ref}
        with _stage_wall(run,folder,'validation-'+job['identity_sha256'],wall_identity,max_job) as job_wall:
            for plan in val.plans:
                if not plan.query_eligible:continue
                if job_wall.exhausted:raise DownstreamError('validation_cumulative_job_wall_limit_exhausted')
                dest=folder/(plan.participant_id+'-'+plan.session_id)
                fitted,fit_ref,timing,candidate=_target_once(ctx,catalog,run,job,plan,dest/'fit',per,'validation')
                episode=val.materialize(plan,device='cpu' if method in {'B4','B5'} else 'cuda:0')
                pred_path=dest/'prediction/receipt.json'
                if pred_path.exists():pred_ref=descriptor(root,pred_path)
                else:
                    with _deadline(min(per,run.wall.remaining_seconds)),torch.autocast(device_type='cuda',enabled=False):
                        prediction=_production_predict(ctx,job,plan,fit_ref,candidate)
                        torch.cuda.synchronize()
                    pred_ref=_commit_prediction(ctx,job,plan,prediction,fit_ref,candidate,dest/'prediction')
                arrays=_prediction_arrays(ctx,pred_ref,job,plan,candidate,fit_ref)
                row=_scores(episode,arrays);row.update(fallback_reason=fitted.fallback_reason,
                    calibration_wall_seconds=timing['calibration_wall_seconds'])
                immutable_json(dest/'score.json',row);rows.append(row)
                records.append({'plan_sha256':plan.temporal_audit_sha256,'candidate':descriptor(root,dest/'fit/candidate.json'),
                    'fit':fit_ref,'timing':descriptor(root,dest/'fit/timing.json'),'prediction':pred_ref,'score':descriptor(root,dest/'score.json')})
                run.guard()
            if job_wall.exhausted:raise DownstreamError('validation_cumulative_finalization_overrun')
            summary=participant_equal_metrics(rows,expected_participants=val.partition_people)
    finally:run.close()
    value={'schema_version':SCHEMA,'status':'SOURCE_EXTENSION_VALIDATION_COMPLETE','synthetic':False,'arm':ctx.arm,
        'job':job,'context':ctx.production_identity,'runtime':runtime,'profile':profile_item,'configuration_receipt':selection_ref,
        'catalog':_catalog_binding(val),'records':records,'sessions':rows,**summary,
        'calibration_median_seconds':statistics.median(r['calibration_wall_seconds'] for r in rows),
        'selected_meta_outer_update':_dependency(ctx,catalog,method,job['configuration'])[2]['selected_meta_outer_update'],
        'resources':run.snapshot(folder),'stage_wall':descriptor(root,folder/'stage_wall.json'),
        'internal_allowed':False,'external_allowed':False}
    immutable_json(path,value);item=descriptor(root,path);verify_validation(root,item,catalog)
    _update_catalog(root,catalog,'validation',job['identity_sha256'],item)
    return {'status':value['status'],'receipt':item,'actual_query_sessions':len(rows)}


def verify_validation(root,item,catalog):
    root=Path(root).resolve(strict=True);catalog=_catalog(root,catalog);value=_json(checked(root,item));job=value['job']
    if job!=_validation_job(job['method'],job['configuration_index'],job['seed'],job['phase']):raise DownstreamError('registered_actual_validation_job_required')
    ctx,train,val=load_context(root,catalog,job['seed']);_production(ctx)
    resources,claim,budget=_resource_proof(root,value['resources'],ctx.arm)
    expected={'schema_version':SCHEMA,'status':'SOURCE_EXTENSION_VALIDATION_COMPLETE','synthetic':False,'arm':ctx.arm,
        'context':ctx.production_identity,'catalog':_catalog_binding(val),'internal_allowed':False,'external_allowed':False}
    if any(value.get(k)!=v for k,v in expected.items()) or value.get('runtime')!=claim['runtime']:
        raise DownstreamError('real_same_arm_validation_identity_required')
    profile=verify_profile(root,value['profile'],catalog)
    if profile['runtime']!=value['runtime']:raise DownstreamError('validation_same_device_profile_required')
    if job['phase']=='confirmation':
        selected=_verify_configuration(root,value['configuration_receipt'],catalog)
        if selected['configuration_index']!=job['configuration_index']:raise DownstreamError('validation_confirmation_configuration_changed')
    elif value.get('configuration_receipt') is not None:raise DownstreamError('screening_cannot_inherit_outcome_selected_config')
    plans={p.temporal_audit_sha256:p for p in val.plans if p.query_eligible};rows=[];seen=set()
    per=SAFETY_FACTOR*profile['maximum_calibration_seconds'][job['method']]+5
    maximum=min(budget['caps']['validation_per_job'],len(plans)*(per+5)+120)
    wall_identity={'job':job,'context':ctx.production_identity,'profile':value['profile'],
        'budget':resources['budget'],'configuration_receipt':value['configuration_receipt']}
    _verify_stage_wall(root,value['stage_wall'],wall_identity,maximum,resources,'validation-'+job['identity_sha256'])
    for record in value.get('records',[]):
        sha=record['plan_sha256']
        if sha not in plans or sha in seen:raise DownstreamError('complete_unique_future_query_catalog_required')
        plan=plans[sha];episode=val.materialize(plan,device='cpu')
        deps=_dependency(ctx,catalog,job['method'],job['configuration'])[2]
        candidate=_candidate(ctx,job,plan,episode.support,deps,resources['budget'],'validation')
        if _json(checked(root,record['candidate']))!=candidate:raise DownstreamError('validation_fit_candidate_identity_changed')
        fitted,timing=_verify_target(ctx,record['fit'],candidate,resources);checked(root,record['timing'])
        if timing['runtime']!=value['runtime'] or timing['calibration_wall_seconds']>per:
            raise DownstreamError('validation_actual_calibration_overrun_or_device_changed')
        arrays=_prediction_arrays(ctx,record['prediction'],job,plan,candidate,record['fit'])
        row=_scores(episode,arrays);row.update(fallback_reason=fitted.fallback_reason,calibration_wall_seconds=timing['calibration_wall_seconds'])
        if _json(checked(root,record['score']))!=row:raise DownstreamError('committed_query_metrics_changed')
        rows.append(row);seen.add(sha)
    if seen!=set(plans) or rows!=value.get('sessions'):raise DownstreamError('all_frozen_validation_sessions_required')
    summary=participant_equal_metrics(rows,expected_participants=val.partition_people)
    if any(value.get(k)!=v for k,v in summary.items()) or value.get('calibration_median_seconds')!=statistics.median(r['calibration_wall_seconds'] for r in rows):
        raise DownstreamError('participant_equal_metrics_or_real_timing_changed')
    return value


def readiness(root,arm):
    """Read-only software/data gates; never open an internal/external catalog."""
    root=Path(root).resolve(strict=True);result=build_plan(arm,root)
    gaps=[];catalog_path=f'runs/source_extension_downstream/{arm}/launch.json'
    if not local(root,catalog_path,exists=False).is_file():
        gaps.append('actual_mainline_all14_three_seed_development_freeze_and_complete_source_extension_manifest_then_initialize')
    else:
        catalog=_latest_catalog(root,catalog_path)
        try:_primary(root,catalog)
        except (ValueError,KeyError,OSError,TypeError):gaps.append('actual_verified_mainline_all14_three_seed_development_freeze')
        try:_budget(root,catalog)
        except (ValueError,KeyError,OSError,TypeError):gaps.append('actual_profile_derived_fixed_arm_budget')
        for name,keys in [('source_f0',map(str,DEVELOPMENT_SEEDS)),('references',map(str,DEVELOPMENT_SEEDS))]:
            for key in keys:
                if key not in catalog.get(name,{}):gaps.append(name+'_seed_'+key)
        if 'all118' not in catalog.get('profiles',{}):gaps.append('own_actual_118_calibration_profile_after_45_screening_meta_jobs')
        if 'development_freeze' not in catalog.get('final',{}):gaps.append('own_146_validation_55_scientific_meta_and_three_seed_Bstar_freeze')
    return {**result,'status':'PENDING_REAL_PREREQUISITES' if gaps else 'DEVELOPMENT_FREEZE_REFERENCE_PRESENT_REVERIFY_BEFORE_USE',
        'minimum_gaps':gaps,'catalog':catalog_path,'scientific_completion_claim':False,
        'next_command':'python tools/run_source_extension_downstream.py --arm '+arm+' --stage auto-development --execute --user-started-gpu',
        'server_actions':False}


def _source_call_floor(root,arm,seed,output):
    """Physical calls including uncertain dispatches are owned by f0 producer."""
    from . import source_extension_training as extension
    claim=local(root,f'runs/source_extensions/claims/{arm}-seed-{seed}.json',exists=False)
    value=_json(claim) if claim.exists() else {}
    return extension._observed(local(root,output+'/fit',exists=False))+value.get('uncertain_optimizer_calls',0)


def _f0_limits(budget,seed):
    plan=budget['source_plans'][str(seed)];target=plan['optimizer_updates']
    return {'stage':'source_extension_training','arm':budget['arm'],'seed':seed,'plan_sha256':plan['plan_sha256'],
        'max_jobs':1,'max_download_bytes':0,'max_wall_seconds':budget['caps']['f0'][str(seed)],
        'max_gpu_seconds':budget['caps']['f0'][str(seed)],'max_fit_invocations':target+1,'max_observed_optimizer_calls':2*target}


@threadpool_limits.wrap(limits=1)
def prepare_f0(root,catalog,seed,*,execute=False,user_started_gpu=False):
    if seed not in DEVELOPMENT_SEEDS:raise DownstreamError('registered_development_seed_required')
    if not execute:return {**build_plan(_catalog(root,catalog)['arm']),'stage':'f0','seed':seed,'status':'PLANNED_NOT_EXECUTED'}
    if not user_started_gpu:raise PermissionError('execute_and_later_user_started_GPU_both_required')
    _require_owned_stage()
    from . import source_extension_training as extension
    root=Path(root).resolve(strict=True);catalog=_latest_catalog(root,catalog);primary=_primary(root,catalog)
    budget,budget_ref=_budget(root,catalog);plan=budget['source_plans'][str(seed)];runtime=_runtime()
    folder=_owned(root,catalog['arm'],f"runs/source_extension_downstream/{catalog['arm']}/source_f0/seed-{seed}")
    folder.mkdir(parents=True,exist_ok=True);path=folder/'receipt.json'
    target=plan['optimizer_updates'];key='source-f0-'+str(seed)
    limits=_f0_limits(budget,seed)
    immutable_json(folder/'limits.json',limits)
    identity={'arm':catalog['arm'],'seed':seed,'plan_sha256':plan['plan_sha256'],'budget':budget_ref,
        'limits':descriptor(root,folder/'limits.json'),'baseline_f0':_primary_seed(primary,seed),'runtime':runtime,'code':_code(root)}
    if path.exists():
        value=verify_f0_stage(root,descriptor(root,path),catalog);_update_catalog(root,catalog,'source_f0',seed,value['source_f0'])
        return {'status':'ALREADY_COMPLETE_VERIFIED','receipt':descriptor(root,path),'new_fits':0}
    run=ArmRun(root,catalog,runtime);result=None
    try:
        prior=run.state['stages'].get(key)
        if prior is not None and prior.get('identity_sha256')!=digest(identity):raise DownstreamError('source_f0_arm_dispatch_identity_changed')
        if prior is not None and not local(root,f"runs/source_extensions/claims/{catalog['arm']}-seed-{seed}.json",exists=False).is_file():
            raise DownstreamError('dispatched_source_f0_original_trajectory_claim_missing_no_restart')
        run.state['stages'][key]={'kind':'source_f0','identity_sha256':digest(identity),'dispatched':True};run.save()
        calls=_source_call_floor(root,catalog['arm'],seed,plan['output'])
        if calls>2*target:raise DownstreamError('source_f0_physical_call_cap_exceeded')
        run.observe('source_calls',key,calls)
        try:
            with _deadline(run.wall.remaining_seconds):
                result=extension.execute_extension(root,arm=catalog['arm'],seed=seed,
                    extension_manifest=catalog['extension_manifest']['path'],extension_sha256=catalog['extension_manifest']['sha256'],
                    baseline_receipt=_primary_seed(primary,seed)['path'],budget_path=(folder/'limits.json').relative_to(root).as_posix(),
                    user_started_gpu=True,pilot_receipt=budget['source_pilot']['path'],output=plan['output'])
        finally:run.observe('source_calls',key,_source_call_floor(root,catalog['arm'],seed,plan['output']))
        run.guard()
    finally:run.close()
    if result.get('status')!='SOURCE_EXTENSION_F0_COMPLETE':return {'status':'PAUSED_NOT_COMPLETE','source_result':result}
    source_ref=result['receipt'];extension.verify_completed_extension(root,source_ref['path'])
    value={'schema_version':SCHEMA,'status':'SOURCE_EXTENSION_F0_ARM_STAGE_COMPLETE','identity':identity,
        'source_f0':source_ref,'resources':run.snapshot(folder),'internal_allowed':False,'external_allowed':False}
    immutable_json(path,value);verify_f0_stage(root,descriptor(root,path),catalog)
    _update_catalog(root,catalog,'source_f0',seed,source_ref)
    return {'status':value['status'],'receipt':descriptor(root,path),'source_f0':source_ref}


def verify_f0_stage(root,item,catalog):
    from .source_extension_training import verify_completed_extension
    root=Path(root).resolve(strict=True);catalog=_catalog(root,catalog);value=_json(checked(root,item))
    checked(root,value['source_f0']);source=verify_completed_extension(root,value['source_f0']['path']);seed=source['pipeline_seed']
    resources,claim,budget=_resource_proof(root,value['resources'],catalog['arm']);identity=value['identity']
    expected={'arm':catalog['arm'],'seed':seed,'plan_sha256':budget['source_plans'][str(seed)]['plan_sha256'],
        'budget':resources['budget'],'limits':source['budget'],'baseline_f0':_primary_seed(_primary(root,catalog),seed),
        'runtime':claim['runtime'],'code':_code(root)}
    key='source-f0-'+str(seed);calls=source['observed_optimizer_calls']+source['uncertain_optimizer_calls']
    if (value.get('schema_version')!=SCHEMA or value.get('status')!='SOURCE_EXTENSION_F0_ARM_STAGE_COMPLETE'
            or identity!=expected or source['arm']!=catalog['arm'] or seed not in DEVELOPMENT_SEEDS
            or resources['ledger'].get('source_calls',{}).get(key)!=calls
            or resources['ledger'].get('stages',{}).get(key)!={'kind':'source_f0','identity_sha256':digest(identity),'dispatched':True}
            or value.get('internal_allowed') is not False or value.get('external_allowed') is not False):
        raise DownstreamError('same_arm_real_source_f0_and_resource_stage_required')
    if _json(checked(root,source['budget']))!=_f0_limits(budget,seed):raise DownstreamError('fixed_profile_derived_source_f0_budget_required')
    return value


@threadpool_limits.wrap(limits=1)
def prepare_reference_stage(root,catalog,seed,*,execute=False,user_started_gpu=False):
    if not execute:return {**build_plan(_catalog(root,catalog)['arm']),'stage':'reference','seed':seed,'status':'PLANNED_NOT_EXECUTED'}
    if not user_started_gpu:raise PermissionError('execute_and_later_user_started_GPU_both_required')
    _require_owned_stage()
    root=Path(root).resolve(strict=True);catalog=_latest_catalog(root,catalog);runtime=_runtime()
    folder=_owned(root,catalog['arm'],f"runs/source_extension_downstream/{catalog['arm']}/reference_stage/seed-{seed}")
    folder.mkdir(parents=True,exist_ok=True);path=folder/'receipt.json'
    if path.exists():
        value=verify_reference_stage(root,descriptor(root,path),catalog);_update_catalog(root,catalog,'references',seed,value['reference'])
        return {'status':'ALREADY_COMPLETE_VERIFIED','receipt':descriptor(root,path),'new_fits':0}
    run=ArmRun(root,catalog,runtime);key='reference-'+str(seed)
    identity={'arm':catalog['arm'],'seed':seed,'source_f0':catalog['source_f0'][str(seed)],'budget':run.budget_ref,'code':_code(root)}
    try:
        with _stage_wall(run,folder,key,identity,run.budget['caps']['reference']) as wall:
            with _deadline(min(run.wall.remaining_seconds,wall.remaining_seconds)):
                ref=prepare_reference(root,catalog,seed,execute=True)
            if wall.exhausted:raise DownstreamError('reference_cumulative_wall_overrun')
            run.guard()
    finally:run.close()
    value={'schema_version':SCHEMA,'status':'SOURCE_EXTENSION_REFERENCE_ARM_STAGE_COMPLETE','identity':identity,'reference':ref,
        'stage_wall':descriptor(root,folder/'stage_wall.json'),'resources':run.snapshot(folder),'internal_allowed':False,'external_allowed':False}
    immutable_json(path,value);verify_reference_stage(root,descriptor(root,path),catalog);_update_catalog(root,catalog,'references',seed,ref)
    return {'status':value['status'],'receipt':descriptor(root,path),'reference':ref}


def verify_reference_stage(root,item,catalog):
    root=Path(root).resolve(strict=True);catalog=_catalog(root,catalog);value=_json(checked(root,item));identity=value['identity'];seed=identity['seed']
    resources,claim,budget=_resource_proof(root,value['resources'],catalog['arm'])
    expected={'arm':catalog['arm'],'seed':seed,'source_f0':catalog['source_f0'][str(seed)],'budget':resources['budget'],'code':_code(root)}
    if (value.get('schema_version')!=SCHEMA or value.get('status')!='SOURCE_EXTENSION_REFERENCE_ARM_STAGE_COMPLETE'
            or identity!=expected or seed not in DEVELOPMENT_SEEDS or value.get('internal_allowed') is not False or value.get('external_allowed') is not False):
        raise DownstreamError('own_reference_arm_stage_identity_required')
    load_reference(root,value['reference'],catalog,seed)
    _verify_stage_wall(root,value['stage_wall'],identity,budget['caps']['reference'],resources,'reference-'+str(seed))
    return value


def _configuration_value(root,catalog,method):
    if method not in PRIMARY_METHODS:raise DownstreamError('registered_method_required')
    rows=[]
    for index in range(len(registered_grid(method))):
        job=_validation_job(method,index,SEEDS[0],'screening');item=catalog.get('validation',{}).get(job['identity_sha256'])
        if item is None:raise DownstreamError('complete_real_registered_configuration_grid_required')
        value=verify_validation(root,item,catalog);rows.append((index,value,item))
    runtime=rows[0][1]['runtime']
    if any(v['runtime']!=runtime for _,v,_ in rows):raise DownstreamError('configuration_grid_same_device_required')
    index,winner,item=min(rows,key=lambda row:(-row[1]['participant_mean_ba'],row[1]['participant_mean_bce'],
        row[1]['selected_meta_outer_update'] or 0,row[0]))
    return {'schema_version':SCHEMA,'status':'SOURCE_EXTENSION_METHOD_CONFIGURATION_FROZEN','arm':catalog['arm'],
        'method':method,'seed':SEEDS[0],'configuration_index':index,'configuration':configuration(method,index),
        'selected_screening_receipt':item,'screening_receipts':[r for _,_,r in rows],
        'runtime':runtime,'primary_development_freeze':catalog['primary_development_freeze'],'code':_code(root),
        'selection':'participant_BA_desc/balanced_BCE_asc/selected_checkpoint_asc/grid_order',
        'internal_allowed':False,'external_allowed':False,'primary_H1_changed':False}


def _verify_configuration(root,item,catalog):
    catalog=_catalog(root,catalog);value=_json(checked(root,item))
    if value!=_configuration_value(root,catalog,value['method']):raise DownstreamError('immutable_complete_grid_configuration_changed')
    return value


def freeze_configurations(root,catalog,*,execute=False):
    root=Path(root).resolve(strict=True);catalog=_latest_catalog(root,catalog)
    if not execute:return {**build_plan(catalog['arm']),'stage':'select','status':'PLANNED_NOT_EXECUTED'}
    _primary(root,catalog);items={}
    for method in PRIMARY_METHODS:
        value=_configuration_value(root,catalog,method)
        path=_owned(root,catalog['arm'],f"runs/source_extension_downstream/{catalog['arm']}/configurations/{method}.json")
        immutable_json(path,value);items[method]=descriptor(root,path);catalog=_update_catalog(root,catalog,'configurations',method,items[method])
    for method in sorted(LEARNED):
        _meta_selection(root,catalog,method,execute=True);catalog=_latest_catalog(root,catalog)
    return {'status':'SOURCE_EXTENSION_ALL14_CONFIGURATIONS_FROZEN','configurations':items,'internal_allowed':False,'external_allowed':False}


def _development_value(root,catalog):
    _primary(root,catalog);selected=[];screening=[];all_meta=[];runtime=None
    for method in PRIMARY_METHODS:
        config_ref=catalog.get('configurations',{}).get(method)
        configured=_verify_configuration(root,config_ref,catalog);index=configured['configuration_index']
        for i in range(len(registered_grid(method))):
            job=_validation_job(method,i,SEEDS[0],'screening');ref=catalog.get('validation',{}).get(job['identity_sha256'])
            verify_validation(root,ref,catalog);screening.append(ref)
        for seed in DEVELOPMENT_SEEDS:
            job=_validation_job(method,index,seed,'screening' if seed==SEEDS[0] else 'confirmation')
            ref=catalog.get('validation',{}).get(job['identity_sha256']);value=verify_validation(root,ref,catalog)
            if runtime is not None and runtime!=value['runtime']:raise DownstreamError('all_development_jobs_same_device_required')
            runtime=value['runtime']
            selected.append({'method':method,'seed':seed,'configuration_index':index,'configuration':configured['configuration'],
                'receipt':ref,'participant_mean_ba':value['participant_mean_ba'],'participant_mean_bce':value['participant_mean_bce'],
                'calibration_median_seconds':value['calibration_median_seconds'],'role':'meta_validation','measured':True,
                'device_identity_sha256':digest(runtime)})
    for method in sorted(LEARNED):
        engineering=catalog.get('meta_engineering',{}).get(method);verify_meta(root,engineering,catalog)
        selected_config=_verify_configuration(root,catalog['configurations'][method],catalog)['configuration']
        for config in registered_grid(method):
            job=MetaJob(method,config.get('dimension',config.get('rank')),config['radius'])
            ref=catalog.get('meta',{}).get(job.identity);verify_meta(root,ref,catalog);all_meta.append(ref)
        for seed in DEVELOPMENT_SEEDS[1:]:
            job=MetaJob(method,selected_config.get('dimension',selected_config.get('rank')),selected_config['radius'],seed,'confirmation')
            ref=catalog.get('meta',{}).get(job.identity);verify_meta(root,ref,catalog);all_meta.append(ref)
    for seed in DEVELOPMENT_SEEDS:
        verify_f0_stage(root,descriptor(root,_owned(root,catalog['arm'],f"runs/source_extension_downstream/{catalog['arm']}/source_f0/seed-{seed}/receipt.json")),catalog)
        verify_reference_stage(root,descriptor(root,_owned(root,catalog['arm'],f"runs/source_extension_downstream/{catalog['arm']}/reference_stage/seed-{seed}/receipt.json")),catalog)
    ctx,_,_=load_context(root,catalog)
    for method,config in source_bank_configurations():load_source_bank(ctx,catalog['source_banks'][_bank_key(method,config)],method,config)
    if len(screening)!=118 or len(selected)!=42 or len(all_meta)!=55:raise DownstreamError('exact_scientific_campaign_counts_required')
    return {'schema_version':SCHEMA,'status':'SOURCE_EXTENSION_PRE_INTERNAL_DEVELOPMENT_FROZEN','arm':catalog['arm'],
        'primary_development_freeze':catalog['primary_development_freeze'],'extension_manifest':catalog['extension_manifest'],
        'method_order':list(PRIMARY_METHODS),'seeds':list(DEVELOPMENT_SEEDS),'source_f0':catalog['source_f0'],
        'references':catalog['references'],'source_banks':catalog['source_banks'],'configurations':catalog['configurations'],
        'screening_validation_receipts':screening,'selected_jobs':selected,'scientific_meta_receipts':all_meta,
        'strongest_control':choose_strongest_control([r for r in selected if r['method'] in CONTROLS]),
        'scientific_meta_jobs':55,'development_validation_jobs':146,'runtime':runtime,'code':_code(root),
        'internal_data_used_for_selection':False,'external_data_used_for_selection':False,
        'internal_allowed':False,'external_allowed':False,'primary_H1_changed':False}


def freeze_development(root,catalog,*,execute=False):
    root=Path(root).resolve(strict=True);catalog=_latest_catalog(root,catalog)
    if not execute:return {**build_plan(catalog['arm']),'stage':'freeze','status':'PLANNED_NOT_EXECUTED'}
    _require_owned_stage()
    value=_development_value(root,catalog)
    path=_owned(root,catalog['arm'],f"runs/source_extension_downstream/{catalog['arm']}/pre_internal_development_freeze.json")
    immutable_json(path,value);ref=descriptor(root,path);_update_catalog(root,catalog,'final','development_freeze',ref)
    return {'status':value['status'],'receipt':ref,'internal_allowed':False,'external_allowed':False}


def verify_development_freeze(root,item,catalog):
    root=Path(root).resolve(strict=True);catalog=_latest_catalog(root,catalog);value=_json(checked(root,item))
    if value!=_development_value(root,catalog):raise DownstreamError('whole_extension_development_freeze_changed')
    budget,budget_ref=_budget(root,catalog)
    resources=_owned(root,catalog['arm'],f"runs/source_extension_downstream/{catalog['arm']}/controller")
    identity={'schema_version':SCHEMA,'arm':catalog['arm'],'budget':budget_ref,'kind':'all_dispatch_and_verification_wall'}
    if _json(resources/'claim.json')!=identity:raise DownstreamError('bounded_controller_claim_required')
    _closed_wall(_json(resources/'wall.json'),digest(identity),budget['max_total_wall_seconds'])
    return value


def auto_development(root,catalog,*,max_actions=None,execute=False,user_started_gpu=False):
    root=Path(root).resolve(strict=True);catalog=_latest_catalog(root,catalog)
    if not execute:return {**readiness(root,catalog['arm']),'stage':'auto-development','status':'PLANNED_NOT_EXECUTED'}
    if not user_started_gpu:raise PermissionError('execute_and_later_user_started_GPU_both_required')
    if max_actions is not None and (type(max_actions) is not int or max_actions<1):raise DownstreamError('positive_action_pause_required')
    _primary(root,catalog);actions=[];flags={'execute':True,'user_started_gpu':True}
    def perform(function,*args,**kwargs):
        if max_actions is not None and len(actions)>=max_actions:return False
        stages={prepare_f0:'f0',prepare_reference_stage:'reference',prepare_source_banks:'source-bank',
            execute_meta:'meta',execute_profile:'profile',execute_validation:'evaluate',
            freeze_configurations:'select',freeze_development:'freeze'}
        if function in stages:
            result=execute_stage(root,catalog,stages[function],args=args,kwargs=kwargs,
                execute=True,user_started_gpu=True)
        else:result=function(root,catalog,*args,**kwargs)
        actions.append(result)
        if result.get('status') in {'PAUSED_NOT_COMPLETE','BLOCKED'}:return False
        return True
    def paused():return {'status':'PAUSED_NOT_COMPLETE','actions':actions,'internal_allowed':False,'external_allowed':False}
    budget_path=_owned(root,catalog['arm'],f"runs/source_extension_downstream/{catalog['arm']}/budget.json")
    if not budget_path.exists():
        if not perform(prepare_budget,execute=True):return paused()
    _budget(root,catalog)
    for seed in DEVELOPMENT_SEEDS:
        catalog=_latest_catalog(root,catalog)
        if str(seed) not in catalog['source_f0']:
            if not perform(prepare_f0,seed,**flags):return paused()
        catalog=_latest_catalog(root,catalog)
        if str(seed) not in catalog['references']:
            if not perform(prepare_reference_stage,seed,**flags):return paused()
    catalog=_latest_catalog(root,catalog)
    if len(catalog['source_banks'])!=6:
        if not perform(prepare_source_banks,**flags):return paused()
    for method in sorted(LEARNED):
        catalog=_latest_catalog(root,catalog)
        if method not in catalog['meta_engineering']:
            if not perform(execute_meta,engineering_job(method),**flags):return paused()
        for config in registered_grid(method):
            catalog=_latest_catalog(root,catalog);job=MetaJob(method,config.get('dimension',config.get('rank')),config['radius'])
            if job.identity not in catalog['meta']:
                if not perform(execute_meta,job,**flags):return paused()
    catalog=_latest_catalog(root,catalog)
    if 'all118' not in catalog['profiles']:
        if not perform(execute_profile,**flags):return paused()
    for method in PRIMARY_METHODS:
        for index in range(len(registered_grid(method))):
            catalog=_latest_catalog(root,catalog);job=_validation_job(method,index,SEEDS[0],'screening')
            if job['identity_sha256'] not in catalog['validation']:
                if not perform(execute_validation,method,index,SEEDS[0],**flags):return paused()
    catalog=_latest_catalog(root,catalog)
    if len(catalog['configurations'])!=14 or len(catalog['meta_selections'])!=5:
        if not perform(freeze_configurations,execute=True):return paused()
    for method in PRIMARY_METHODS:
        catalog=_latest_catalog(root,catalog);selected=_verify_configuration(root,catalog['configurations'][method],catalog)
        for seed in DEVELOPMENT_SEEDS[1:]:
            catalog=_latest_catalog(root,catalog);config=selected['configuration'];index=selected['configuration_index']
            if method in LEARNED:
                job=MetaJob(method,config.get('dimension',config.get('rank')),config['radius'],seed,'confirmation')
                if job.identity not in catalog['meta']:
                    if not perform(execute_meta,job,**flags):return paused()
            catalog=_latest_catalog(root,catalog);job=_validation_job(method,index,seed,'confirmation')
            if job['identity_sha256'] not in catalog['validation']:
                if not perform(execute_validation,method,index,seed,'confirmation',**flags):return paused()
    catalog=_latest_catalog(root,catalog)
    if not perform(freeze_development,execute=True):return paused()
    return {'status':'SOURCE_EXTENSION_PRE_INTERNAL_DEVELOPMENT_FROZEN','actions':actions,'internal_allowed':False,'external_allowed':False}


def _stage_functions():
    return {'f0':prepare_f0,'reference':prepare_reference_stage,'source-bank':prepare_source_banks,
        'meta':execute_meta,'profile':execute_profile,'evaluate':execute_validation,
        'select':freeze_configurations,'freeze':freeze_development}


def _owned_worker(root,catalog,stage,args,kwargs,connection,owner_pid,deadline):
    """One independent process group; receives authority before any dispatch."""
    from .training_campaign import start_stage_watchdog,live_group_members
    global _OWNED_STAGE
    os.setsid();pgid=os.getpgrp();_OWNED_STAGE=True
    stop,watchdog=start_stage_watchdog(owner_pid,pgid,deadline)
    try:
        connection.send({'type':'owned_group_ready','pid':os.getpid(),'pgid':pgid})
        if connection.recv()!='dispatch_owned_stage':return
        with threadpool_limits(limits=1):result=_stage_functions()[stage](root,catalog,*args,**kwargs)
        connection.send({'type':'result','ok':True,'result':result})
    except BaseException as error:
        connection.send({'type':'result','ok':False,'reason':str(error) if isinstance(error,(DownstreamError,PermissionError)) else 'owned_stage_failed_'+type(error).__name__})
    finally:
        try:connection.recv()
        except (EOFError,OSError):pass
        while any(pid!=os.getpid() for pid in live_group_members(pgid)):stop.wait(.1)
        stop.set();watchdog.join(.2);connection.close()


def execute_stage(root,catalog,stage,*,args=(),kwargs=None,execute=False,user_started_gpu=False):
    """Hard deadline plus lost-owner cleanup; never starts a server or network."""
    if stage not in _stage_functions():raise DownstreamError('registered_development_only_stage_required')
    kwargs=dict(kwargs or {})
    if not execute:
        kwargs['execute']=False
        return _stage_functions()[stage](root,catalog,*args,**kwargs)
    if not user_started_gpu:raise PermissionError('execute_and_later_user_started_GPU_both_required')
    root=Path(root).resolve(strict=True);catalog=_latest_catalog(root,catalog);budget,budget_ref=_budget(root,catalog)
    _primary(root,catalog);kwargs['execute']=True
    if stage not in {'select','freeze'}:kwargs['user_started_gpu']=True
    resources=_owned(root,catalog['arm'],f"runs/source_extension_downstream/{catalog['arm']}/controller")
    resources.mkdir(parents=True,exist_ok=True)
    lock=(resources/'owner.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:lock.close();raise DownstreamError('single_extension_controller_already_active') from None
    identity={'schema_version':SCHEMA,'arm':catalog['arm'],'budget':budget_ref,'kind':'all_dispatch_and_verification_wall'}
    sha=digest(identity);wall=None;worker=None;parent=None;child=None;pgid=None;started=False;result=None
    try:
        if (resources/'claim.json').exists() and not (resources/'wall.json').is_file():raise DownstreamError('controller_wall_missing_no_reset')
        immutable_json(resources/'claim.json',identity);floor=0
        for path in (resources/'snapshots').glob('*.json'):
            value=_json(path)
            if path.stem!=digest(value):raise DownstreamError('controller_snapshot_changed')
            floor=max(floor,_closed_wall(value,sha,budget['max_total_wall_seconds']))
        wall=PersistentWallBudget(resources/'wall.json',sha,budget['max_total_wall_seconds'],checkpoint_wall_seconds=floor)
        wall.start_interval()
        if wall.exhausted:raise DownstreamError('controller_cumulative_wall_exhausted')
        caps=budget['caps'];stage_limit=wall.remaining_seconds
        if stage=='f0':stage_limit=min(stage_limit,caps['f0'][str(args[0])])
        elif stage=='reference':stage_limit=min(stage_limit,caps['reference'])
        elif stage=='source-bank':stage_limit=min(stage_limit,6*caps['source_bank_per_job'])
        elif stage=='meta':
            job=args[0];stage_limit=min(stage_limit,caps['meta_engineering' if job.phase=='engineering' else 'meta_science_per_job'][job.method])
        elif stage=='profile':stage_limit=min(stage_limit,caps['calibration_profile'])
        elif stage=='evaluate':stage_limit=min(stage_limit,caps['validation_per_job'])
        if stage_limit<=5:raise DownstreamError('fixed_stage_or_controller_wall_exhausted')
        deadline=time.monotonic()+stage_limit
        from .training_campaign import stop_owned_stage
        ctx=multiprocessing.get_context('spawn');parent,child=ctx.Pipe(duplex=True)
        worker=ctx.Process(target=_owned_worker,args=(str(root),catalog,stage,args,kwargs,child,os.getpid(),deadline))
        worker.start();started=True;child.close()
        while time.monotonic()<deadline:
            if parent.poll(min(.1,max(0,deadline-time.monotonic()))):
                try:message=parent.recv()
                except EOFError:raise DownstreamError('owned_stage_exited_preserve_same_dispatch_and_checkpoint') from None
                if message.get('type')=='owned_group_ready':
                    if pgid is not None or message.get('pid')!=worker.pid or message.get('pgid')!=worker.pid:
                        raise DownstreamError('owned_stage_group_handshake_invalid')
                    pgid=worker.pid
                    if time.monotonic()>=deadline:break
                    parent.send('dispatch_owned_stage')
                elif message.get('type')=='result' and pgid is not None:
                    if time.monotonic()>=deadline:break
                    if message.get('ok') is not True:raise DownstreamError(message.get('reason','owned_stage_failed'))
                    result=message['result'];parent.send('result_received');break
                else:raise DownstreamError('stage_result_without_owned_process_authority')
            if not worker.is_alive():raise DownstreamError('owned_stage_exited_preserve_same_dispatch_and_checkpoint')
        if result is None:raise DownstreamError('hard_stage_deadline_preserve_same_dispatch_and_checkpoint')
    finally:
        try:
            if started:
                from .training_campaign import stop_owned_stage
                stop_owned_stage(worker,pgid)
            if wall is not None:
                wall.graceful_close();wall.close();value=_json(resources/'wall.json')
                _closed_wall(value,sha,budget['max_total_wall_seconds'])
                immutable_json(resources/'snapshots'/(digest(value)+'.json'),value)
        finally:
            if child is not None:child.close()
            if parent is not None:parent.close()
            lock.close()
    result=dict(result);result['controller_closed_wall']=descriptor(root,resources/'snapshots'/(digest(value)+'.json'))
    result['server_actions']=False;return result
