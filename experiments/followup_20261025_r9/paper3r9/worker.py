"""Frozen-feature R9 GP evaluation with per-session resume and train-only reference."""
import argparse
from dataclasses import asdict, dataclass, replace
import fcntl
import os
from pathlib import Path
import platform
import random
import signal
import threading
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from p3work.spatial_geometry import joint_trial_rms
from paper3r7.data import Catalog, load_source
from paper3r7.baseline import fit_reference, reference_logits
from paper3r7.diagnostics import perturb_episode
from paper3r7.io import atomic, digest, emit, put, read, sha
from .gp import GPConfig, fit_gp, fit_reference_geometry, ReferenceGeometry

@dataclass(frozen=True)
class Config:
    kernel: str = 'rbf'
    tau: float = .3
    lengthscale: float = 1.
    predictive: str = 'anchored_laplace'
    anchor_cap: float = .75
    preprocessing: str = 'raw'
    jitter: float = 1e-8
    gh_points: int = 20
    temperature: float = 1.

class Pause(Exception):
    pass

def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value

def json_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: json_tree(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [json_tree(v) for v in value]
    return value

def identity_for(plan, task, source, data, dependency=None):
    task = dict(task)
    if 'external_manifest' in task:
        task['external_manifest'] = {'manifest_sha256': data.get('manifest_sha256')}
    return digest({'schema': 'r9-task-v1', 'recipe': plan['recipe_sha256'],
        'task': task, 'source': source, 'data': data, 'dependency': dependency})

def summarize(rows):
    if not rows:
        raise ValueError('No evaluable sessions')
    people = {}
    for row in rows:
        people.setdefault(row['participant_id'], []).append(row)
    values = [{'participant_id': p, 'ba': float(np.mean([r['ba'] for r in rs])),
               'bce': float(np.mean([r['bce'] for r in rs]))}
              for p,rs in sorted(people.items())]
    return {'participant_mean_ba': float(np.mean([r['ba'] for r in values])),
        'participant_mean_bce': float(np.mean([r['bce'] for r in values])),
        'participants': values, 'sessions': len(rows),
        'fallback_sessions': sum(bool(r.get('fallback')) for r in rows)}

def _batches(function, x, batch_size=64):
    return torch.cat([function(x[start:start+batch_size]) for start in range(0, len(x), batch_size)])

def balanced_loss(logits, y):
    weights = logits.new_empty(len(y))
    for c in (0,1):
        mask = y == c
        if not bool(mask.any()):
            raise ValueError('Both query classes are required')
        weights[mask] = .5 / mask.sum()
    return (weights * F.cross_entropy(logits, y, reduction='none')).sum()

def fit_head(source, support, steps):
    """Original head-only SGD with explicitly controlled iteration budget."""
    state={'variant':'head_only','fallback_reason':None}
    diagnostics={'optimizer_updates':0,'head_steps_requested':steps,'uses_support_labels':True}
    if not all(bool((support.y==c).any()) for c in (0,1)):
        state['fallback_reason']='insufficient_class_support'
        return state,diagnostics,state['fallback_reason']
    from p3work.spatial_geometry import joint_trial_rms
    with torch.no_grad():features,base=source.features_and_logits(joint_trial_rms(support.x))
    dw=torch.zeros(source.feature_dim,device=features.device,dtype=features.dtype,requires_grad=True)
    db=torch.zeros((),device=features.device,dtype=features.dtype,requires_grad=True)
    for _ in range(steps):
        delta=features@dw+db
        logits=base+torch.stack((-delta/2,delta/2),1)
        loss=balanced_loss(logits,support.y)+.001*(dw.square().sum()+db.square())/(source.feature_dim+1)
        gw,gb=torch.autograd.grad(loss,(dw,db))
        dw=(dw-.01*gw).detach().requires_grad_(True)
        db=(db-.01*gb).detach().requires_grad_(True)
    state.update(delta_w=dw.detach(),delta_b=db.detach())
    diagnostics['optimizer_updates']=steps
    return state,diagnostics,None

def perturb(episode, geometry, session, scenario):
    if scenario != 'support_label_permutation':
        return perturb_episode(episode,geometry,session,scenario)
    seed=int(digest(['r9-support-label-permutation-v1',session.participant_id,
                     session.session_id])[:15],16)
    generator=torch.Generator(device='cpu').manual_seed(seed)
    order=torch.randperm(len(episode.support.y),generator=generator).to(episode.support.y.device)
    support=replace(episode.support,y=episode.support.y[order])
    values=dict(vars(episode));values['support']=support
    return SimpleNamespace(**values),{'synthetic_stress':True,'scenario':scenario,
        'scope':'support_labels_only_count_preserving','support_changed':True,
        'query_changed':False,'support_signal_changed':False,'labels_changed':True,
        'direction_seed':seed,'class_counts_preserved':True,
        'label_permutation_indices':order.cpu().tolist()}

def evaluate(source, geometry, catalog, config, params, folder, stop, progress,
             *, budget=20, scenario='clean', variant='transport', partition='holdout',
             source_identity=None, batch_size=64):
    folder = Path(folder)
    parameter_id = digest({'source': source_identity, 'data': catalog.identity,
        'config': asdict(config), 'parameters': json_tree({k:v for k,v in params.items() if not k.startswith('_')} if isinstance(params,dict) else params),
        'variant': variant, 'partition': partition, 'budget': budget, 'scenario': scenario,
        'schema': 'r9-session-v1'})
    rows = []
    device = geometry.q.device
    for index, session in enumerate(catalog.plans):
        if stop.is_set():
            raise Pause()
        progress.update(status='EVALUATING', stage=partition, session=index,
            sessions_total=len(catalog.plans), budget=budget, scenario=scenario)
        path = folder / f'{session.participant_id}-{session.session_id}.json'
        expected_uids = list(session.audit['query_uids'])
        expected_labels = [session.rows[u]['label'] for u in expected_uids]
        signature = digest({'uids': expected_uids, 'labels': expected_labels})
        row = read(path) if path.exists() else None
        if row is not None and not (row.get('parameter_id') == parameter_id
                and row.get('query_signature') == signature
                and row.get('event_uids') == expected_uids and row.get('labels') == expected_labels):
            row = None
        if row is None:
            episode = catalog.materialize(session, device, budget=budget)
            episode, perturbation = perturb(episode, geometry, session, scenario)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            started = time.monotonic()
            fitted = None
            try:
                runtime_params=params
                if variant=='gp' and params is not None:
                    context_id=digest({'source':source_identity,'data':catalog.identity,
                        'participant':session.participant_id,'session':session.session_id,
                        'budget':budget,'scenario':scenario,'preprocessing':config.preprocessing,
                        'query_signature':signature,'support_ids':list(episode.support.event_uids),
                        'acquired_ids':list(episode.support.acquired_event_uids),
                        'dtype':'float64','schema':'r9-frozen-feature-cache-v1'})
                    runtime_params=dict(params,_feature_context=Path(params['_feature_cache_root'])/context_id) if params.get('_feature_cache_root') else params
                fitted, diagnostics, fallback = _fitter(source, geometry, episode.support,
                                                       config, runtime_params, variant)
                predict = _predictor(source, geometry, fitted, variant)
            except (FloatingPointError, torch.linalg.LinAlgError) as error:
                fallback = f'numeric_session_failure:{type(error).__name__}:{error}'
                diagnostics = {'numeric_fallback': True, 'uses_future_query_for_fit': False}
                predict = source.source_logits
            if isinstance(fitted,dict) and 'gp' in fitted:
                atomic(path.with_suffix('.gp.pt'),lambda p:torch.save({
                    'gp':fitted['gp'].state_dict(),'reference':fitted['reference'].state_dict(),
                    'ea':cpu_tree(fitted['ea']),'config':asdict(fitted['config']),
                    'parameter_id':parameter_id,'query_labels_used_for_fit':False},p))
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            calibration_seconds = time.monotonic() - started
            started = time.monotonic()
            with torch.no_grad():
                logits = _batches(predict, episode.query_x, batch_size)
                if isinstance(fitted,dict) and fitted.get('feature_context') is not None:
                    cursor=[0]
                    def original_source_predict(x):
                        start=cursor[0];cursor[0]+=len(x)
                        name=(f'query-{start:06d}-{len(x):04d}.pt' if config.preprocessing=='raw'
                              else f'original-source-{start:06d}-{len(x):04d}.pt')
                        return feature_logits(source,geometry,x,None,fitted['feature_context']/name)[1]
                    base=_batches(original_source_predict,episode.query_x,batch_size)
                else:
                    base = _batches(source.source_logits, episode.query_x, batch_size)
                if not bool(torch.isfinite(logits).all() and torch.isfinite(base).all()):
                    raise FloatingPointError('Nonfinite future-query prediction')
                y = episode.query_y
                pred, base_pred = logits.argmax(1), base.argmax(1)
                if not all(bool((y==c).any()) for c in (0,1)):
                    raise ValueError('Both future-query classes are required')
                balanced = lambda m: .5 * sum(float(m[y==c].double().mean()) for c in (0,1))
                prob, base_prob = logits.double().softmax(1), base.double().softmax(1)
                logs, base_logs = logits.double().log_softmax(1), base.double().log_softmax(1)
                bce = float(balanced_loss(logits.double(), y))
                f0_bce = float(balanced_loss(base.double(), y))
                brier = balanced((prob[:,1]-y).square())
                f0_brier = balanced((base_prob[:,1]-y).square())
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            prediction_seconds = time.monotonic() - started
            bins=[]
            predinfo=fitted.get('prediction_diagnostics',{}) if isinstance(fitted,dict) else {}
            if predinfo.get('information_fraction'):
                information=torch.tensor(predinfo['information_fraction'],device=y.device,dtype=torch.float64)
                anchor=torch.tensor(predinfo['anchor_reference_probability1'],device=y.device,dtype=torch.float64)
                for lower,upper in zip((0.,.01,.05,.1,.2,.4,.7),( .01,.05,.1,.2,.4,.7,1.0000001)):
                    for cls in (0,1):
                        mask=(information>=lower)&(information<upper)&(y==cls)
                        if not bool(mask.any()):continue
                        trueprob=prob[:,cls]
                        anchortrue=anchor if cls==1 else 1-anchor
                        bins.append({'information_lower':lower,'information_upper':min(1.,upper),
                            'class':cls,'query_trials':int(mask.sum()),
                            'adapted_correct':int((pred[mask]==y[mask]).sum()),
                            'source_correct':int((base_pred[mask]==y[mask]).sum()),
                            'negative_flips':int(((base_pred[mask]==y[mask])&(pred[mask]!=y[mask])).sum()),
                            'corrected_errors':int(((base_pred[mask]!=y[mask])&(pred[mask]==y[mask])).sum()),
                            'adapted_nll_sum':float(-trueprob[mask].clamp_min(1e-15).log().sum()),
                            'source_nll_sum':float(-base_prob[mask,cls].clamp_min(1e-15).log().sum()),
                            'anchor_reference_nll_sum':float(-anchortrue[mask].clamp_min(1e-15).log().sum()),
                            'information_sum':float(information[mask].sum()),
                            'anchor_certified_trials':sum(bool(v) for v,keep in zip(predinfo['anchor_decision_preservation_certificate'],mask.detach().cpu().tolist()) if keep)})
            row = {'parameter_id': parameter_id, 'source_identity': source_identity,
                'data_identity': catalog.identity, 'partition': partition,
                'participant_id': session.participant_id, 'session_id': session.session_id,
                'budget': budget, 'scenario': scenario, 'variant': variant,
                'ba': balanced(pred==y), 'bce': bce, 'nll_balanced': bce,
                'f0_ba': balanced(base_pred==y), 'f0_nll_balanced': f0_bce,
                'brier_balanced': brier, 'f0_brier_balanced': f0_brier,
                'negative_flip_rate': balanced((base_pred==y)&(pred!=y)),
                'corrected_error_rate': balanced((base_pred!=y)&(pred==y)),
                'fallback': fallback is not None, 'fallback_reason': fallback,
                'query_mechanism_bins':bins, 'diagnostics': diagnostics, 'prediction_diagnostics': fitted.get('prediction_diagnostics',{}) if isinstance(fitted,dict) else {}, 'calibration_seconds': calibration_seconds,
                'prediction_seconds': prediction_seconds,
                'prediction_seconds_per_trial': prediction_seconds/len(y),
                'supportdiagnostics_seconds': 0., 'diagnostics_seconds': 0.,
                'event_uids': list(episode.query_ids), 'run_ids': list(episode.run_ids),
                'labels': y.cpu().tolist(), 'predictions': pred.cpu().tolist(),
                'probabilities': prob.cpu().tolist(), 'log_probabilities': logs.cpu().tolist(),
                'f0_predictions': base_pred.cpu().tolist(), 'f0_probabilities': base_prob.cpu().tolist(),
                'f0_log_probabilities': base_logs.cpu().tolist(), 'query_signature': signature,
                'perturbation': perturbation, 'order_scope': session.audit.get('order_scope'),
                'temporal_risk_eligible': session.audit.get('temporal_risk_eligible', True),
                'support_acquired_trials': len(episode.support.acquired_event_uids),
                'support_valid_trials': len(episode.support.y)}
            put(path, row)
        rows.append(row)
        progress.update(session=index+1)
    if stop.is_set():
        raise Pause()
    result = summarize(rows)
    result.update(parameter_id=parameter_id, partition=partition, budget=budget, scenario=scenario,
        data_identity=catalog.identity, expected_sessions=[{'participant_id':r['participant_id'],
        'session_id':r['session_id'],'query_signature':r['query_signature']} for r in rows])
    put(folder / 'summary.json', result)
    return result

@torch.no_grad()
def feature_logits(source, geometry, x, ea=None, cache_path=None):
    if cache_path is not None:
        cache_path=Path(cache_path)
        if cache_path.is_file():
            try:
                cached=torch.load(cache_path,map_location='cpu',weights_only=True)
                if cached['features'].shape[0]==len(x):
                    return cached['features'].to(x.device),cached['base_logits'].to(x.device)
            except (OSError,RuntimeError,ValueError,EOFError,KeyError):pass
    if ea is not None and ea.get('fallback_reason') is None:
        x = ea['transform'].to(x.device) @ x.double()
    if not len(x):
        return x.new_empty((0,source.feature_dim)), x.new_empty((0,2))
    f, base = source.features_and_logits(joint_trial_rms(x))
    f,base=f.double(),base.double()
    if cache_path is not None:
        atomic(cache_path,lambda p:torch.save({'features':f.cpu(),'base_logits':base.cpu()},p))
    return f,base


def _fitter(source, geometry, support, config, params, variant):
    if variant == 'source_frozen': return None, {}, None
    if variant=='source_temperature':return {'temperature':config.temperature},{'temperature':config.temperature,'label_free_temperature_control':True},None
    if variant=='ea_temperature':
        state,diagnostics=fit_reference(source,geometry,support,'support_EA')
        state['temperature']=config.temperature
        diagnostics.update(temperature=config.temperature,label_free_temperature_control=True)
        return state,diagnostics,diagnostics.get('fallback_reason')
    if variant in ('head_sgd10', 'head_sgd100'):
        return fit_head(source, support, 10 if variant == 'head_sgd10' else 100)
    if variant == 'support_EA':
        state, diagnostics = fit_reference(source, geometry, support, variant)
        return state, diagnostics, diagnostics.get('fallback_reason')
    ea = None
    if config.preprocessing == 'ea':
        ea, _ = fit_reference(source, geometry, support, 'support_EA')
    reference = ReferenceGeometry.from_state_dict(params['reference'])
    context=params.get('_feature_context')
    cache=None if context is None else context/'support.pt'
    features, base = feature_logits(source, geometry, support.x, ea, cache)
    gpconfig = GPConfig(kernel=config.kernel, tau=config.tau,
        lengthscale=config.lengthscale * float(params.get('reference_lengthscale', 1.)),
        jitter=config.jitter, predictive=config.predictive if config.predictive != 'constant_mixture' else 'laplace',
        gh_points=config.gh_points, anchor_cap=config.anchor_cap)
    fitted = fit_gp(reference.transform(features), base, support.y, gpconfig)
    state = {'gp': fitted, 'reference': reference, 'ea': ea, 'config': config, 'feature_context':context, 'query_cursor':0}
    diagnostics = dict(fitted.diagnostics, preprocessing=config.preprocessing,
        target_head_fitted=False, source_encoder_frozen=True,
        uses_future_query_for_fit=False, hyperparameters_from=params.get('selection_scope','fixed_predeclared'),
        requested_tau=config.tau, requested_lengthscale_multiplier=config.lengthscale)
    state['prediction_diagnostics'] = {}
    state['diagnostics'] = diagnostics
    reason = diagnostics.get('fallback_reason')
    return state, diagnostics, reason


def _predictor(source, geometry, fitted, variant):
    if variant == 'source_frozen': return source.source_logits
    if variant=='source_temperature':return lambda x:source.source_logits(x)/fitted['temperature']
    if variant=='ea_temperature':return lambda x:reference_logits(source,geometry,fitted,x)/fitted['temperature']
    if variant in ('support_EA', 'head_sgd10', 'head_sgd100'):
        return lambda x: reference_logits(source, geometry, fitted, x)
    def predict(x):
        context=fitted.get('feature_context')
        start=fitted['query_cursor'];fitted['query_cursor']+=len(x)
        cache=None if context is None else context/f'query-{start:06d}-{len(x):04d}.pt'
        f, base = feature_logits(source, geometry, x, fitted['ea'],cache)
        cfg = fitted['config']
        mode = 'laplace' if cfg.predictive == 'constant_mixture' else cfg.predictive
        white = fitted['reference'].transform(f)
        probability = fitted['gp'].predict_proba(white, base, predictive=mode)
        bad=~torch.isfinite(white).all(1)
        clean=torch.where(bad[:,None],torch.zeros_like(white),white)
        shift, variance, information, prior = fitted['gp'].posterior_moments(clean)
        bad=bad|~(torch.isfinite(shift)&torch.isfinite(variance)&torch.isfinite(information)&torch.isfinite(prior))
        shift=torch.where(bad,torch.zeros_like(shift),shift)
        variance=torch.where(bad,torch.zeros_like(variance),variance)
        information=torch.where(bad,torch.zeros_like(information),information)
        prior=torch.where(bad,torch.zeros_like(prior),prior)
        fitted['diagnostics']['nonfinite_query_feature_fallback_rows']=fitted['diagnostics'].get('nonfinite_query_feature_fallback_rows',0)+int(bad.sum())
        data = fitted['prediction_diagnostics']
        if fitted['gp'].diagnostics.get('fallback_reason'):
            applied = torch.zeros_like(information)
        elif cfg.predictive == 'anchored_laplace':
            applied = cfg.anchor_cap*information
        elif cfg.predictive == 'constant_mixture':
            applied = torch.full_like(information,cfg.anchor_cap)
        else:
            applied = torch.ones_like(information)
        for name, value in (('residual_mean',shift),('posterior_variance',variance),
                ('prior_variance',prior),('information_fraction',information),
                ('anchor_weight',cfg.anchor_cap*information),('applied_gp_mixture_weight',applied)):
            data.setdefault(name,[]).extend(value.detach().cpu().tolist())
            fitted['diagnostics']['query_mean_'+name] = float(np.mean(data[name])) if data[name] else 0.
        fitted['diagnostics']['query_max_anchor_weight'] = max(data.get('anchor_weight',[]) or [0.])
        data.setdefault('anchor_reference_probability1',[]).extend(base.softmax(1)[:,1].detach().cpu().tolist())
        data.setdefault('unanchored_probability1',[]).extend(probability[:,1].detach().cpu().tolist())
        if cfg.predictive == 'constant_mixture':
            probability = (1-cfg.anchor_cap)*base.softmax(1) + cfg.anchor_cap*probability
        certificate=(base.softmax(1)[:,1]-.5).abs()>applied
        data.setdefault('anchor_decision_preservation_certificate',[]).extend(certificate.cpu().tolist())
        fitted['diagnostics']['query_anchor_certified_fraction']=float(np.mean(data['anchor_decision_preservation_certificate']))
        fitted['diagnostics']['query_anchor_certificate_violations']=fitted['diagnostics'].get('query_anchor_certificate_violations',0)+int((certificate&(probability.argmax(1)!=base.argmax(1))).sum())
        return probability.double().clamp_min(1e-15).log()
    return predict


def _prefix(catalog, session, device, budget=20):
    """Read paid prefix only: no future query materialization for reference geometry."""
    from p3work.adapt import CalibrationPrefix
    acquired = tuple(session.audit['acquired_prefix_uids'][:budget])
    ids = tuple(u for u in acquired if session.rows[u]['valid'])
    array = catalog.array(session)
    x = np.array(array[[session.rows[u]['cache_index'] for u in ids]], dtype=np.float64)
    y = [session.rows[u]['label'] for u in ids]
    return CalibrationPrefix(torch.from_numpy(x).to(device),
        torch.tensor(y, device=device, dtype=torch.long), ids, acquired, catalog.signal_sha,
        'meta_train_support')


def _reference_state(source, geometry, catalog, preprocessing, folder, stop, progress, identity):
    people = {}
    rows = []
    for i, session in enumerate(catalog.plans):
        if stop.is_set(): raise Pause()
        progress.update(status='PREPARING_REFERENCE', stage='meta_train_prefix_features',
            session=i, sessions_total=len(catalog.plans))
        path = folder/'reference-sessions'/f'{session.participant_id}-{session.session_id}.pt'
        saved = None
        if path.is_file():
            try:
                value = torch.load(path, map_location='cpu', weights_only=True)
                if value.get('identity') == identity: saved = value
            except (OSError, RuntimeError, ValueError, EOFError): pass
        if saved is None:
            support = _prefix(catalog, session, geometry.q.device)
            ea = fit_reference(source, geometry, support, 'support_EA')[0] if preprocessing=='ea' else None
            features, _ = feature_logits(source, geometry, support.x, ea)
            saved = {'identity':identity, 'features':features.cpu(),
                'participant_id':session.participant_id, 'session_id':session.session_id,
                'support_event_uids':list(support.event_uids),
                'support_acquired_trials':len(support.acquired_event_uids)}
            atomic(path,lambda p:torch.save(saved,p))
        rows.append(saved)
        if len(saved['features']): people.setdefault(session.participant_id, []).append(saved)
    if not people: raise ValueError('No valid meta_train support features')
    features, weights = [], []
    for person, sessions in sorted(people.items()):
        for row in sessions:
            value = row['features'].to(geometry.q.device)
            features.append(value)
            weights.append(torch.full((len(value),), 1/(len(people)*len(sessions)*len(value)),
                dtype=torch.float64, device=geometry.q.device))
    provenance = {'role':'meta_train_prefix20_only', 'data_identity':catalog.identity,
        'preprocessing':preprocessing, 'source_identity':identity,
        'weighting':'equal_person_equal_session_equal_valid_prefix_trial',
        'participants':len(people), 'sessions':sum(map(len,people.values())),
        'future_queries_read':False}
    reference = fit_reference_geometry(torch.cat(features), weights=torch.cat(weights), provenance=provenance)
    state = {'identity':identity, 'reference':reference.state_dict(),
        'reference_lengthscale':float(reference.lengthscale), 'provenance':provenance}
    path = folder/'selected-reference.pt'
    atomic(path,lambda p:torch.save(cpu_tree(state),p))
    return {'selection':None, 'outer_updates':0, 'holdout_used_for_selection':False,
        'selected_checkpoint':{'path':str(path.resolve()), 'sha256':sha(path)},
        'reference_scope':provenance}


def _dependency(plan, task, key, expected_family, source_identity):
    declared = next((t for t in plan['tasks'] if t['key']==key), None)
    if Path(key).name != key or key not in task.get('requires', []) or declared is None:
        raise ValueError('Undeclared dependency')
    folder = Path(plan['output'])/key
    result = read(folder/'result.json')
    if (result.get('status')!='COMPLETE' or result.get('task')!=declared
            or result.get('recipe_sha256')!=plan['recipe_sha256']
            or declared['family']!=expected_family or result.get('source_identity')!=source_identity):
        raise ValueError('Dependency scientific identity differs')
    path = Path(result['selected_checkpoint']['path'])
    if not path.is_absolute(): path=Path(plan['output'])/path
    if not path.resolve().is_relative_to(folder.resolve()) or sha(path)!=result['selected_checkpoint']['sha256']:
        raise ValueError('Dependency checkpoint differs')
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['identity']!=result['identity']: raise ValueError('Dependency state identity differs')
    return state, {'key':key, 'identity':state['identity'], 'sha256':sha(path)}


def _catalog(plan, task, family):
    if family=='feature_reference': return Catalog(plan['root'],'meta_train',cache_gb=4)
    if family=='select': return Catalog(plan['root'],'meta_validation',cache_gb=4)
    if task.get('external_manifest'):
        if read(task['external_manifest']).get('dataset_id')=='Pan2023':
            from paper3r7.pan2023 import PanCatalog
            return PanCatalog(plan['root'],task['external_manifest'],cache_gb=2)
        from paper3r7.external import ExternalCatalog
        return ExternalCatalog(plan['root'],task['external_manifest'],cache_gb=2)
    return Catalog(plan['root'],'internal_holdout',cache_gb=4)


def run(args):
    plan=read(args.plan); task=plan['tasks'][args.task]; key=task['key']
    if Path(key).name!=key or key in ('.','..'): raise ValueError('Unsafe task key')
    folder=Path(plan['output'])/key; folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close(); emit('ALREADY_RUNNING',task=key); return 0
    stop, done=threading.Event(),threading.Event()
    if os.environ.get('PAPER3R9_STOP_REQUESTED')=='1': stop.set()
    for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:stop.set())
    progress={'task':key,'status':'STARTING','stage':'loading','outer_update':0}
    def beat():
        while not done.is_set():
            current=dict(progress,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            emit('TRAIN_PROGRESS',**current)
            try: put(folder/'progress.json',current)
            except OSError as error: emit('PROGRESS_WRITE_RETRY',reason=str(error))
            done.wait(30)
    thread=threading.Thread(target=beat,daemon=True);thread.start()
    started=time.monotonic()
    try:
        if stop.is_set(): raise Pause()
        device=torch.device(args.device)
        threads=min(16,max(1,int(os.environ.get('SLURM_CPUS_PER_TASK','16'))));torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        seed=int(task['seed']);torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
        source,geometry,source_identity=load_source(plan['root'],seed,device)
        family=task['family'];catalog=_catalog(plan,task,family)
        if source_identity['signal_contract_sha256']!=catalog.signal_sha: raise ValueError('Signal contract differs')
        params=None;dependency={};cfg=dict(task.get('config',{}))
        if task.get('reference_key'):
            params,dependency['reference']=_dependency(plan,task,task['reference_key'],'feature_reference',source_identity)
        if task.get('selection_key'):
            selected,dependency['selection']=_dependency(plan,task,task['selection_key'],'select',source_identity)
            cfg.update({k:selected['selected_config'][k] for k in ('tau','lengthscale')})
            params=dict(params,selection_scope='meta_validation_participant_mean_balanced_NLL')
        if params is not None: params=dict(params,_feature_cache_root=str(Path(plan['output'])/'frozen-feature-cache'/plan['recipe_sha256']))
        config=Config(**cfg)
        identity=identity_for(plan,task,source_identity,catalog.identity,dependency or None)
        if (folder/'result.json').exists():
            previous=read(folder/'result.json')
            if previous.get('status')=='COMPLETE':
                if previous.get('identity')!=identity: raise ValueError('Completed task identity differs')
                emit('COMPLETED_TASK_REUSED',task=key);progress.update(status='COMPLETE',stage='done');return 0
        put(folder/'runtime.json',{'python':platform.python_version(),'torch':str(torch.__version__),
            'numpy':np.__version__,'cuda_build':torch.version.cuda,
            'device':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
            'cpus_per_worker':threads,'source_frozen':True,'compute_node_downloads':False,
            'full_test_or_data_rescan':False})
        if family=='feature_reference':
            result=_reference_state(source,geometry,catalog,config.preprocessing,folder,stop,progress,identity)
        elif family=='select':
            history=[]
            for index,candidate in enumerate(task['candidates']):
                if stop.is_set(): raise Pause()
                candidate_config=replace(config,**candidate)
                progress.update(stage='meta_validation_hyperparameters',candidate=index,candidates_total=len(task['candidates']))
                score=evaluate(source,geometry,catalog,candidate_config,params,folder/f'validation/candidate-{index:03d}',
                    stop,progress,partition='meta_validation',variant='gp',source_identity=source_identity)
                item={'candidate':candidate,'score':score,'index':index};history.append(item)
                put(folder/'selection-progress.json',{'identity':identity,'candidates':history})
            chosen=min(history,key=lambda h:(h['score']['participant_mean_bce'],-h['score']['participant_mean_ba'],h['index']))
            selected_config=asdict(replace(config,**chosen['candidate']))
            selected_path=folder/'selected-hyperparameters.pt'
            atomic(selected_path,lambda p:torch.save({'identity':identity,'selected_config':selected_config,'selection':chosen},p))
            result={'selection':chosen,'selection_history':history,'selected_config':selected_config,
                'selected_checkpoint':{'path':str(selected_path.resolve()),'sha256':sha(selected_path)},
                'outer_updates':0,'holdout_used_for_selection':False,
                'selection_metric':'meta_validation_participant_mean_balanced_NLL',
                'matched_MAP_PP_hyperparameters':True}
        else:
            for budget in task.get('budgets',[20]):
                for scenario in task.get('scenarios',['clean']):
                    evaluate(source,geometry,catalog,config,params,folder/f'evaluations/{task["partition"]}/B{budget}/{scenario}',
                        stop,progress,budget=budget,scenario=scenario,variant=task.get('variant','gp'),
                        partition=task['partition'],source_identity=source_identity)
            result={'selection':None,'outer_updates':0,'dependency':dependency,
                'holdout_used_for_selection':False,'effective_config':asdict(config)}
        result.update(status='COMPLETE',recipe_sha256=plan['recipe_sha256'],identity=identity,task=task,
            source_identity=source_identity,data_identity=catalog.identity,elapsed_seconds=time.monotonic()-started)
        put(folder/'result.json',result);(folder/'error.json').unlink(missing_ok=True)
        progress.update(status='COMPLETE',stage='done');emit('TASK_COMPLETE',task=key);return 0
    except Pause:
        progress.update(status='PAUSED_CHECKPOINT_SAVED');emit('PAUSED_CHECKPOINT_SAVED',task=key)
        return 75
    except Exception as error:
        failure={'status':'FAILED','task':key,'error_type':type(error).__name__,'reason':str(error),'traceback':traceback.format_exc()}
        print(failure['traceback'],flush=True)
        try:put(folder/'error.json',failure)
        except OSError:pass
        progress.update(status='FAILED',reason=str(error));return 1
    finally:
        done.set();thread.join(timeout=2)
        try:put(folder/'progress.json',dict(progress,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        except OSError:pass
        lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--task',required=True,type=int)
    parser.add_argument('--device',default='cuda:0')
    return run(parser.parse_args())

if __name__=='__main__':raise SystemExit(main())
