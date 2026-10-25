"""Round 8 frozen-source transport experiments with durable episode boundaries.

Support fitting never receives future query inputs or labels. Meta-training uses
future query labels only for the pre-existing meta_train participants. Numerical
fallbacks are recorded per session; failure does not delete prior results.
"""
import argparse
import copy
from dataclasses import asdict, replace
import fcntl
import os
from pathlib import Path
import pickle
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

from paper3r7.data import Catalog, load_source
from paper3r7.baseline import fit_reference, reference_logits
from paper3r7.diagnostics import perturb_episode
from paper3r7.io import atomic, digest, emit, put, read, sha
from .transport import TransportConfig, fit_transport, predict_transport, init_meta_parameters


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


def save_checkpoint(folder, state):
    folder = Path(folder)
    pointer = folder / 'latest.json'
    old = read(pointer) if pointer.exists() else {}
    generation = old.get('generation', 0) + 1
    path = folder / f'state-{generation % 2}.pt'
    payload = cpu_tree(state)
    payload['rng_cpu'] = torch.get_rng_state()
    payload['rng_cuda'] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    payload['rng_python'] = random.getstate()
    n = np.random.get_state()
    payload['rng_numpy'] = (n[0], n[1].tolist(), n[2], n[3], n[4])
    atomic(path, lambda temp: torch.save(payload, temp))
    put(pointer, {'path': path.name, 'generation': generation, 'sha256': sha(path),
        'outer_update': state['update'],
        'previous': {k:v for k,v in old.items() if k != 'previous'}})


def load_checkpoint(folder, identity, *, restore_rng=True):
    folder = Path(folder)
    pointer = read(folder / 'latest.json')
    for entry in (pointer, pointer.get('previous', {})):
        name = entry.get('path', '')
        if not name or Path(name).name != name:
            continue
        path = folder / name
        if not path.is_file() or sha(path) != entry.get('sha256'):
            continue
        try:
            state = torch.load(path, map_location='cpu', weights_only=True)
        except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError):
            continue
        if state.get('identity') != identity:
            raise ValueError('Checkpoint belongs to a different experiment')
        if restore_rng:
            torch.set_rng_state(state['rng_cpu'])
            if torch.cuda.is_available() and state.get('rng_cuda'):
                torch.cuda.set_rng_state_all(state['rng_cuda'])
            random.setstate(state['rng_python'])
            n = state['rng_numpy']
            np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
        if entry is not pointer:
            put(folder / 'latest.json', entry)
        return state
    raise IOError('Neither checkpoint generation is readable; files retained')


def identity_for(plan, task, source, data, dependency=None):
    task = dict(task)
    if 'external_manifest' in task:
        task['external_manifest'] = {'manifest_sha256': data.get('manifest_sha256')}
    return digest({'schema': 'r8-task-v1', 'recipe': plan['recipe_sha256'],
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


def _fitter(source, geometry, support, config, params, variant):
    if variant == 'source_frozen':
        return None, {}, None
    if variant in ('head_sgd10','head_sgd100'):
        return fit_head(source,support,10 if variant=='head_sgd10' else 100)
    if variant in ('support_EA', 'head_only'):
        state, metadata = fit_reference(source, geometry, support, variant)
        return state, metadata, metadata.get('fallback_reason')
    fitted = fit_transport(source, geometry, support, config, meta_parameters=params)
    return fitted, dict(fitted.diagnostics), fitted.fallback_reason


def _predictor(source, geometry, fitted, variant):
    if variant == 'source_frozen':
        return source.source_logits
    if variant in ('support_EA', 'head_only','head_sgd10','head_sgd100'):
        return lambda x: reference_logits(source, geometry, fitted, x)
    return lambda x: predict_transport(source, geometry, fitted, x)


def perturb(episode, geometry, session, scenario):
    if scenario != 'support_label_permutation':
        return perturb_episode(episode,geometry,session,scenario)
    seed=int(digest(['r8-support-label-permutation-v1',session.participant_id,
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
        'config': asdict(config), 'parameters': json_tree(params),
        'variant': variant, 'partition': partition, 'budget': budget, 'scenario': scenario,
        'schema': 'r8-session-v1'})
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
            try:
                fitted, diagnostics, fallback = _fitter(source, geometry, episode.support,
                                                       config, params, variant)
                predict = _predictor(source, geometry, fitted, variant)
            except (FloatingPointError, torch.linalg.LinAlgError) as error:
                fallback = f'numeric_session_failure:{type(error).__name__}:{error}'
                diagnostics = {'numeric_fallback': True, 'uses_future_query_for_fit': False}
                predict = source.source_logits
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            calibration_seconds = time.monotonic() - started
            started = time.monotonic()
            with torch.no_grad():
                logits = _batches(predict, episode.query_x, batch_size)
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
                'diagnostics': diagnostics, 'calibration_seconds': calibration_seconds,
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


def train_step(source, geometry, catalog, config, parameters, optimizer, *, seed,
               update, episodes_per_update=4, batch_size=64):
    """Balanced future loss with chunked backprop; no full query graph is retained."""
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.
    for episode_index in range(episodes_per_update):
        episode = catalog.materialize(catalog.draw(seed, update*episodes_per_update+episode_index),
                                      geometry.q.device, budget=20)
        fitted = fit_transport(source, geometry, episode.support, config, meta_parameters=parameters)
        if fitted.fallback_reason is not None:
            # A legitimate support-empty fallback has zero parameter gradient.
            # Retain its loss contribution rather than resampling a different person.
            pass
        y = episode.query_y
        if not all(bool((y==c).any()) for c in (0,1)):
            raise ValueError('Meta-training query must contain both classes')
        weights = torch.empty(len(y), device=y.device, dtype=torch.float64)
        for c in (0,1):
            weights[y==c] = .5 / (y==c).sum()
        chunks = list(range(0, len(y), batch_size))
        for part, start in enumerate(chunks):
            end = start+batch_size
            pred = predict_transport(source, geometry, fitted, episode.query_x[start:end])
            loss = (weights[start:end] * F.cross_entropy(pred.double(), y[start:end], reduction='none')).sum()
            loss_total += float(loss.detach()) / episodes_per_update
            # +0 preserves valid all-source fallback as a zero gradient update.
            objective = loss/episodes_per_update + parameters.sum()*0
            objective.backward(retain_graph=part+1 < len(chunks))
    if parameters.grad is None or not bool(torch.isfinite(parameters.grad).all()):
        raise FloatingPointError('Nonfinite meta-parameter gradient')
    grad_norm = float(parameters.grad.norm())
    torch.nn.utils.clip_grad_norm_([parameters], 10.)
    optimizer.step()
    if not bool(torch.isfinite(parameters).all()):
        raise FloatingPointError('Nonfinite updated meta parameters')
    return {'outer_query_nll': loss_total, 'meta_gradient_norm': grad_norm}


def _load_dependency(plan, task, key, family, device, source_identity):
    if Path(key).name != key or key not in task.get('requires', []):
        raise ValueError('Missing declared model dependency')
    folder=Path(plan['output'])/key
    result=read(folder/'result.json')
    declared=next((t for t in plan['tasks'] if t['key']==key),None)
    if (result.get('status')!='COMPLETE' or result.get('recipe_sha256')!=plan['recipe_sha256']
            or declared is None or declared!=result.get('task') or declared.get('family')!=family
            or declared['seed']!=task['seed'] or result.get('source_identity')!=source_identity):
        raise ValueError('Model dependency identity differs')
    locator=result['selected_checkpoint'];path=Path(locator['path'])
    if not path.is_absolute():path=Path(plan['output'])/path
    if not path.resolve().is_relative_to(folder.resolve()) or sha(path)!=locator['sha256']:
        raise ValueError('Selected model dependency content differs')
    selected=torch.load(path,map_location='cpu',weights_only=True)
    if selected.get('identity')!=result['identity']:
        raise ValueError('Selected model dependency identity differs')
    return selected,{'key':key,'identity':result['identity'],'sha256':locator['sha256']}


def load_meta(plan, task, device, source_identity):
    learned=reference=None;deps={}
    if task.get('meta_key'):
        model,deps['learned']=_load_dependency(plan,task,task['meta_key'],'train',device,source_identity)
        learned=model['parameters'].to(device)
    if task.get('reference_key'):
        model,deps['class_reference']=_load_dependency(plan,task,task['reference_key'],
                                                      'class_reference',device,source_identity)
        reference=model['class_reference'].to(device)
    params={'learned':learned,'class_reference':reference} if reference is not None else learned
    return params,deps or None


def class_reference(source, geometry, catalog, folder, stop, progress, identity):
    """Equal-person then equal-session class contrast from paid meta_train prefixes.

    Each class covariance first averages per-trial unit-trace native CAR
    covariances. No target evaluation session or meta-validation row is used.
    """
    folder=Path(folder)
    people={};excluded=[]
    for index,session in enumerate(catalog.plans):
        if stop.is_set():raise Pause()
        progress.update(status='PREPARING_CLASS_REFERENCE',stage='meta_train_prefixes',
                        session=index,sessions_total=len(catalog.plans),budget=20)
        path=folder/'class-reference-sessions'/f'{session.participant_id}-{session.session_id}.json'
        row=read(path) if path.exists() else None
        if row is None or row.get('identity')!=identity:
            from p3work.adapt import CalibrationPrefix
            acquired=tuple(session.audit['acquired_prefix_uids'][:20])
            ids=tuple(u for u in acquired if session.rows[u]['valid'])
            a=catalog.array(session)
            xs=np.array(a[[session.rows[u]['cache_index'] for u in ids]],dtype=np.float64)
            labels=[session.rows[u]['label'] for u in ids]
            support=CalibrationPrefix(torch.from_numpy(xs).to(geometry.q.device),
                torch.tensor(labels,device=geometry.q.device,dtype=torch.long),
                ids,acquired,catalog.signal_sha,'meta_train_support')
            if not all(bool((support.y==c).any()) for c in (0,1)):
                row={'identity':identity,'participant_id':session.participant_id,
                     'session_id':session.session_id,'excluded':'missing_support_class'}
            else:
                z=geometry.q@support.x.double();cov=z@z.transpose(-1,-2)
                cov=cov/cov.diagonal(dim1=-2,dim2=-1).sum(-1)[:,None,None]
                contrast=cov[support.y==1].mean(0)-cov[support.y==0].mean(0)
                row={'identity':identity,'participant_id':session.participant_id,
                     'session_id':session.session_id,'contrast':contrast.cpu().tolist(),
                     'support_event_uids':list(support.event_uids),
                     'support_acquired_trials':len(support.acquired_event_uids)}
            put(path,row)
        if row.get('excluded'):excluded.append(row)
        else:people.setdefault(session.participant_id,[]).append(torch.tensor(row['contrast'],
                                   dtype=geometry.q.dtype,device=geometry.q.device))
        progress.update(session=index+1)
    if not people:raise ValueError('No two-class meta-training prefixes for class reference')
    contrast=torch.stack([torch.stack(rows).mean(0) for _,rows in sorted(people.items())]).mean(0)
    reference=geometry.rref_invsqrt@contrast@geometry.rref_invsqrt
    reference=(reference+reference.T)/2
    path=folder/'selected-class-reference.pt'
    atomic(path,lambda temp:torch.save({'identity':identity,'class_reference':cpu_tree(reference)},temp))
    return {'selection':None,'outer_updates':0,'holdout_used_for_selection':False,
        'selected_checkpoint':{'path':str(path.resolve()),'sha256':sha(path)},
        'reference_scope':'meta_train_first20_acquired_prefix_only',
        'class_contrast_weighting':'equal_participants_equal_sessions_equal_trials_within_class',
        'reference_participants':len(people),'reference_sessions':sum(map(len,people.values())),
        'excluded_reference_sessions':excluded}


def run(args):
    plan = read(args.plan)
    task = plan['tasks'][args.task]
    key = task['key']
    if Path(key).name != key or key in ('.','..'):
        raise ValueError('Task key must name one directory')
    folder = Path(plan['output'])/key
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder/'.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        emit('ALREADY_RUNNING', task=key)
        return 0
    stop, done = threading.Event(), threading.Event()
    if os.environ.get('PAPER3R8_STOP_REQUESTED') == '1':
        stop.set()
    for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:stop.set())
    progress = {'task':key,'status':'STARTING','stage':'loading','outer_update':0}
    def beat():
        while not done.is_set():
            current = dict(progress,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            emit('TRAIN_PROGRESS',**current)
            try: put(folder/'progress.json',current)
            except OSError as error: emit('PROGRESS_WRITE_RETRY',reason=str(error))
            done.wait(30)
    heartbeat = threading.Thread(target=beat,daemon=True)
    heartbeat.start()
    state = None
    params = None
    optimizer = None
    started = time.monotonic()
    prior_elapsed = 0.
    def save():
        if state is not None:
            state.update(parameters=cpu_tree(params),optimizer=cpu_tree(optimizer.state_dict()),
                elapsed_seconds=prior_elapsed+time.monotonic()-started)
            save_checkpoint(folder,state)
    try:
        if stop.is_set(): raise Pause()
        device = torch.device(args.device)
        if device.type=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('Allocated CUDA device unavailable')
        threads = min(16,max(1,int(os.environ.get('SLURM_CPUS_PER_TASK','16'))))
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        seed = int(task['seed'])
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        config = TransportConfig(**task.get('config',{}))
        variant = task.get('variant','transport')
        source, geometry, source_identity = load_source(plan['root'],seed,device)
        family = task.get('family','evaluation')
        progress.update(stage='catalog_loading')
        if family=='class_reference':
            catalog=Catalog(plan['root'],'meta_train',cache_gb=4)
            dependency=None
        elif family=='train':
            if config.parameterization=='fixed':
                raise ValueError('Meta-training requires global or evidence parameters')
            catalog = Catalog(plan['root'],'meta_train',cache_gb=4)
            validation = Catalog(plan['root'],'meta_validation',cache_gb=2)
            dependency = None
            params = init_meta_parameters(config,device=device).detach().requires_grad_(True)
        else:
            if task.get('external_manifest') or family=='transfer':
                manifest = task.get('external_manifest',plan.get('external_manifest'))
                if read(manifest).get('dataset_id')=='Pan2023':
                    from paper3r7.pan2023 import PanCatalog
                    catalog=PanCatalog(plan['root'],manifest,cache_gb=2)
                else:
                    from paper3r7.external import ExternalCatalog
                    catalog=ExternalCatalog(plan['root'],manifest,cache_gb=2)
            else:
                catalog = Catalog(plan['root'],'internal_holdout',cache_gb=4)
            params, dependency = load_meta(plan,task,device,source_identity)
            if config.parameterization!='fixed' and params is None:
                raise ValueError('Learned evaluation requires completed meta_key')
        if source_identity['signal_contract_sha256'] != catalog.signal_sha:
            raise ValueError('Source and evaluation signal contracts differ')
        identity = identity_for(plan,task,source_identity,catalog.identity,dependency)
        previous = read(folder/'result.json') if (folder/'result.json').exists() else None
        if previous and previous.get('status')=='COMPLETE':
            if previous.get('identity')!=identity:
                raise ValueError('Completed task identity differs')
            emit('COMPLETED_TASK_REUSED',task=key)
            progress.update(status='COMPLETE',stage='done',outer_update=previous.get('outer_updates',0))
            return 0
        runtime = {'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,
            'cuda_build':torch.version.cuda,'device':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
            'cpus_per_worker':threads,'source_frozen':True,'full_test_or_data_rescan':False}
        put(folder/'runtime.json',runtime)
        if family=='class_reference':
            result=class_reference(source,geometry,catalog,folder,stop,progress,identity)
            result['elapsed_seconds']=time.monotonic()-started
        elif family=='train':
            total=int(task.get('updates',plan.get('updates',1000)))
            every=int(task.get('validate_every',plan.get('validate_every',100)))
            check=int(task.get('checkpoint_every',plan.get('checkpoint_every',10)))
            episodes=int(task.get('episodes_per_update',4))
            batch=int(task.get('query_batch_size',64))
            if min(total,every,check,episodes,batch)<1:
                raise ValueError('Positive iteration and batch counts required')
            optimizer=torch.optim.Adam([params],lr=float(task.get('meta_lr',.01)))
            if (folder/'latest.json').exists():
                state=load_checkpoint(folder,identity)
                with torch.no_grad(): params.copy_(state['parameters'].to(device))
                optimizer.load_state_dict(state['optimizer'])
                emit('CHECKPOINT_RESUMED',task=key,update=state['update'])
            else:
                state={'identity':identity,'update':0,'validated_update':-1,'history':[],
                       'best_score':None,'elapsed_seconds':0.}
                save()
            prior_elapsed=state.get('elapsed_seconds',0.)
            started=time.monotonic()
            while True:
                update=state['update']
                if stop.is_set(): save(); raise Pause()
                if (update==0 or update%every==0 or update==total) and state['validated_update']<update:
                    save()
                    score=evaluate(source,geometry,validation,config,params.detach(),
                        folder/f'validation/{update:04d}',stop,progress,partition='meta_validation',
                        source_identity=source_identity,batch_size=batch)
                    score['outer_update']=update
                    choose=lambda s:(s['participant_mean_bce'],-s['participant_mean_ba'],s['outer_update'])
                    if state['best_score'] is None or choose(score)<choose(state['best_score']):
                        state.update(best_score=score,best_parameters=cpu_tree(params))
                    state['history'].append(score);state['validated_update']=update
                    save()
                if update>=total: break
                progress.update(stage='meta_train',status='TRAINING',outer_update=update,total_updates=total)
                before_parameters=params.detach().clone()
                before_optimizer=copy.deepcopy(optimizer.state_dict())
                try:
                    metrics=train_step(source,geometry,catalog,config,params,optimizer,seed=seed,update=update,
                        episodes_per_update=episodes,batch_size=batch)
                except Exception:
                    # A failed numerical update must not overwrite the latest
                    # durable complete update with half-applied optimizer state.
                    with torch.no_grad():params.copy_(before_parameters)
                    optimizer.load_state_dict(before_optimizer)
                    raise
                state.update(update=update+1,metrics=metrics)
                progress.update(outer_update=update+1,**metrics)
                if state['update']%check==0:save()
            selected_path=folder/'selected-parameters.pt'
            atomic(selected_path,lambda temp:torch.save({'identity':identity,
                'parameters':state['best_parameters'],'config':asdict(config)},temp))
            save()
            result={'selection':state['best_score'],'outer_updates':state['update'],
                'selected_checkpoint':{'path':str(selected_path.resolve()),'sha256':sha(selected_path)},
                'holdout_used_for_selection':False,'selection_metric':'participant_mean_nll',
                'elapsed_seconds':state['elapsed_seconds']}
        else:
            partition=task.get('partition','holdout')
            for budget in task.get('budgets',plan.get('budgets',[20,4,8,12,16])):
                for scenario in task.get('scenarios',['clean']):
                    if scenario!='clean' and budget!=20:continue
                    evaluate(source,geometry,catalog,config,params,
                        folder/f'evaluations/{partition}/B{budget}/{scenario}',stop,progress,
                        budget=budget,scenario=scenario,variant=variant,partition=partition,
                        source_identity=source_identity,batch_size=int(task.get('query_batch_size',64)))
            result={'selection':None,'outer_updates':0,'dependency':dependency,
                'holdout_used_for_selection':False,'elapsed_seconds':time.monotonic()-started}
        result.update(status='COMPLETE',recipe_sha256=plan['recipe_sha256'],identity=identity,
                      task=task,source_identity=source_identity,data_identity=catalog.identity)
        put(folder/'result.json',result)
        (folder/'error.json').unlink(missing_ok=True)
        progress.update(status='COMPLETE',stage='done',outer_update=result['outer_updates'])
        emit('TASK_COMPLETE',task=key)
        return 0
    except Pause:
        save()
        progress.update(status='PAUSED_CHECKPOINT_SAVED')
        emit('PAUSED_CHECKPOINT_SAVED',task=key,outer_update=state['update'] if state else 0)
        return 75
    except Exception as error:
        try:save()
        except Exception as checkpoint_error:emit('CHECKPOINT_WRITE_FAILED',reason=str(checkpoint_error))
        failure={'status':'FAILED','task':key,'error_type':type(error).__name__,'reason':str(error),
                 'traceback':traceback.format_exc()}
        print(failure['traceback'],flush=True)
        try:put(folder/'error.json',failure)
        except OSError:pass
        progress.update(status='FAILED',reason=str(error))
        return 1
    finally:
        done.set();heartbeat.join(timeout=2)
        try:put(folder/'progress.json',dict(progress,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        except OSError:pass
        lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--task',required=True,type=int)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
