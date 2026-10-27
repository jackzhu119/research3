"""Persistent M12 scientific blocks with durable, label-separated episode seals.

Monitoring failures never control a fit. Only externally requested graceful
stops are observed between complete session/budget units; raw fit checkpoints
are retained. No process signals, scheduler calls or network access occur here.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import csv
from dataclasses import is_dataclass, replace
import io
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

from .common import atomic_bytes, atomic_json, digest, lock, read_json, sha_file, utc
from .metrics import balanced_support_bce, probabilities, session_metrics

_PROVIDERS = OrderedDict()
_VERIFIED = OrderedDict()
_CACHE_LIMIT = 4096
CONTROL_KINDS = frozenset(('freeze_tau', 'pilot_decision', 'development_freeze', 'final_report'))


class PauseRequested(Exception):
    pass


class ScientificArtifactError(ValueError):
    pass


def _jsonable(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _np(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def _npz(path, **arrays):
    target = io.BytesIO()
    np.savez_compressed(target, **arrays)
    atomic_bytes(path, target.getvalue())


def _csv(path, rows):
    target = io.StringIO(newline='')
    fields = sorted({k for row in rows for k in row})
    writer = csv.DictWriter(target, fields, lineterminator='\n'); writer.writeheader()
    for row in rows:
        writer.writerow({key: (json.dumps(value, sort_keys=True, allow_nan=False)
                              if isinstance(value, (dict, list, tuple)) else value)
                         for key, value in row.items()})
    atomic_bytes(path, target.getvalue().encode())


def _rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def _output(plan):
    path = Path(plan['output'])
    return (path if path.is_absolute() else Path(plan['root'])/path).resolve()


def _folder(plan, task):
    key = task['key']
    if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]+', key) or key in ('.', '..'):
        raise ValueError('Concrete path-safe scientific task key required')
    return _output(plan)/'tasks'/key


def _stop(stop):
    requested = stop.is_set() if hasattr(stop, 'is_set') else bool(stop() if callable(stop) else stop)
    if requested:
        raise PauseRequested('Graceful pause between durable scientific units')


def _notice(folder, task, callback, **fields):
    record = dict(task_key=task['key'], task_kind=task.get('kind'), updated_utc=utc(), **_jsonable(fields))
    # This is intentionally independent from durable checkpoints and receipts.
    try:
        atomic_json(folder/'progress.json', record)
    except Exception:
        pass
    if callback is not None:
        try:
            callback(**record)
        except Exception:
            pass


def _descriptor(path, output):
    path = Path(path).resolve(); output = Path(output).resolve()
    if not path.is_relative_to(output) or path.is_symlink() or not path.is_file():
        raise ScientificArtifactError('Scientific output descriptor escaped declared run')
    return dict(path=path.relative_to(output).as_posix(), bytes=path.stat().st_size, sha256=sha_file(path))


def _verify_files(output, descriptors):
    output = Path(output).resolve()
    if not isinstance(descriptors, list) or not descriptors:
        raise ScientificArtifactError('Scientific receipt has no declared output files')
    for item in descriptors:
        value = Path(item['path'])
        if value.is_absolute() or '..' in value.parts:
            raise ScientificArtifactError('Scientific receipt path escaped declared run')
        path = (output/value).resolve()
        if not path.is_relative_to(output) or path.is_symlink() or not path.is_file():
            raise ScientificArtifactError('Scientific receipt file missing or escaped')
        info = path.stat(); stamp = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        key = (str(path), item['sha256'], item['bytes'])
        if _VERIFIED.get(key) == stamp:
            _VERIFIED.move_to_end(key); continue
        if info.st_size != item['bytes'] or sha_file(path) != item['sha256']:
            raise ScientificArtifactError('Committed scientific output SHA or size changed: '+str(value))
        _VERIFIED[key] = stamp
        while len(_VERIFIED) > _CACHE_LIMIT:
            _VERIFIED.popitem(last=False)


def _existing_result(plan, task):
    path = _folder(plan, task)/'result.json'
    if not path.exists():
        return None
    value = read_json(path)
    if (not isinstance(value, dict) or value.get('status') != 'COMPLETE'
            or value.get('task_key') != task['key'] or value.get('task_identity') != digest(task)
            or value.get('recipe_sha256') != plan.get('recipe_sha256')):
        raise ScientificArtifactError('Existing scientific result identity/status differs')
    _verify_files(_output(plan), value.get('files', value.get('artifacts')))
    return value


def _seal(plan, task, result, paths=()):
    output = _output(plan); descriptors = list(result.get('files', result.get('artifacts', [])))
    descriptors += [_descriptor(path, output) for path in paths]
    unique = {}
    for item in descriptors:
        if item['path'] in unique and unique[item['path']] != item:
            raise ScientificArtifactError('Same scientific output declared with different bytes')
        unique[item['path']] = item
    files = [unique[key] for key in sorted(unique)]
    _verify_files(output, files)
    result = dict(_jsonable(result), status='COMPLETE', task=task, task_key=task['key'],
                  task_identity=digest(task), files=files, artifacts=files,
                  recipe_sha256=plan.get('recipe_sha256'), completed_utc=utc(),
                  synthetic_fixture=bool(plan.get('synthetic_fixture', False)))
    atomic_json(_folder(plan, task)/'result.json', result)
    return result


def _provider(plan, device, factory=None):
    if factory is not None:
        if plan.get('synthetic_fixture') is not True:
            raise ValueError('Provider fixtures require an explicitly synthetic plan')
        return factory(plan, device)
    key = (str(Path(plan['root']).resolve()), str(_output(plan)), str(device),
           digest(plan.get('inputs_descriptor', plan.get('native_inputs', plan.get('inputs', {})))))
    if key not in _PROVIDERS:
        from .native_data import DataProvider
        _PROVIDERS[key] = DataProvider(plan, device)
        while len(_PROVIDERS) > 2:
            _PROVIDERS.popitem(last=False)
    _PROVIDERS.move_to_end(key)
    return _PROVIDERS[key]


def _config(plan, task, session=None, budget=None, intervention=None):
    config = copy.deepcopy(plan.get('config', {}))
    if not isinstance(config, dict):
        raise ValueError('Frozen scientific config must be an object')
    config['episode_identity'] = dict(role=task.get('role'), participant=task.get('participant'),
        session=str(getattr(session, 'session_id', 'reference')), budget=budget,
        source_seed=int(task['seed']), intervention=intervention or 'clean')
    return config


def _tau(plan, budget):
    value = plan.get('tau_by_budget')
    if value is None:
        path = Path(plan.get('frozen_tau_path', _output(plan)/'frozen_tau.json'))
        if not path.is_absolute():
            path = _output(plan)/path
        sealed = read_json(path)
        if not isinstance(sealed, dict):
            raise ScientificArtifactError('Support-only tau freeze is not complete')
        value = sealed.get('tau_by_budget', sealed.get('tau'))
    if not isinstance(value, dict) or str(budget) not in value:
        raise ScientificArtifactError('Frozen tau_B missing for registered budget')
    result = float(value[str(budget)])
    if not math.isfinite(result) or result < 0:
        raise ScientificArtifactError('Frozen tau_B must be finite and nonnegative')
    return result


def _unit_folder(folder, session, budget, intervention='clean'):
    token = digest([str(session.participant_id), str(session.session_id), session.identity])[:20]
    return folder/'units'/('session-'+token)/('B'+str(budget))/intervention


def _raw_cache(folder):
    result = {}
    for path in sorted((folder/'raw-fits').glob('*.json')):
        value = read_json(path)
        if not isinstance(value, dict) or value.get('name') is None or not isinstance(value.get('state'), dict):
            raise ScientificArtifactError('Raw scientific checkpoint is malformed')
        result[value['name']] = value['state']
    def checkpoint(name, state):
        if not isinstance(name, str) or not isinstance(state, dict):
            raise TypeError('Raw checkpoint needs a named JSON state')
        atomic_json(folder/'raw-fits'/(digest(name)+'.json'), {'name': name, 'state': _jsonable(state)})
        result[name] = state
    return result, checkpoint


def _unit_receipt(unit, identity, output):
    path = unit/'receipt.json'
    if not path.exists():
        return None
    value = read_json(path)
    if (not isinstance(value, dict) or value.get('status') != 'COMPLETE'
            or value.get('unit_identity') != identity):
        raise ScientificArtifactError('Committed session-budget identity differs')
    _verify_files(output, value['files'])
    return value


def _replace_x(view, x):
    if is_dataclass(view):
        return replace(view, x=x)
    names = ('x', 'y', 'event_uids', 'acquired_event_uids', 'signal_sha', 'role')
    return SimpleNamespace(**{name: x if name == 'x' else getattr(view, name)
                              for name in names if name == 'x' or hasattr(view, name)})


def _base_method(name):
    if name == 'support_ea':
        return 'support_EA', None
    for base in ('random_q', 'drop_random'):
        found = re.fullmatch(base+r'_(\d+)', name)
        if found:
            return base, int(found.group(1))
    head = re.fullmatch(r'random_q_(\d+)_head_(frozen|bias|full)', name)
    if head:
        return name, int(head.group(1))
    return name, None


def _metric_layer(state):
    layer = state.get('layer', 'reference_rms')
    if layer != 'anchor_fallback':
        return layer
    method = state.get('method', '')
    if method == 'support_ea':
        return 'anchor'
    if method == 'full_sgd':
        return 'raw'
    if method.startswith('drop_'):
        return 'removal_reference_rms'
    if method in ('weak_q', 'top_q_weak_match'):
        return 'weak_reference_rms'
    return 'reference_rms'


def _support_score(algorithm, model, state, support):
    predicted = algorithm.predict(model, state, support.x)
    return balanced_support_bce(_np(support.y), predicted)


def _reuse_clean_shift_fit(plan, task, session, budget, support, model, reference):
    """Reuse only sealed support-derived states, never target metric selection."""
    for candidate in plan.get('tasks', []):
        if (candidate.get('kind') != 'paired_block' or candidate.get('role') != 'pilot'
                or str(candidate.get('participant')) != str(task['participant'])
                or int(candidate.get('seed', -1)) != int(task['seed'])):
            continue
        source_result = _existing_result(plan, candidate)
        if source_result is None:
            continue
        unit = _unit_folder(_folder(plan, candidate), session, budget)
        receipt = read_json(unit/'receipt.json')
        if not isinstance(receipt, dict):
            continue
        identity = receipt.get('unit_identity', {})
        expected = dict(task_identity=digest(candidate), recipe_sha256=plan.get('recipe_sha256'),
            session_identity=session.identity, budget=int(budget),
            support_event_uids=list(support.event_uids), acquired_event_uids=list(support.acquired_event_uids),
            source_identity=_jsonable(getattr(model, 'identity', {})),
            reference_identity=_jsonable(reference['identity']), fixed_tau=_tau(plan, budget),
            intervention='clean')
        if any(identity.get(key) != value for key, value in expected.items()):
            raise ScientificArtifactError('Clean support-derived cache differs from known-shift control')
        _unit_receipt(unit, identity, _output(plan))
        source_path = unit/'states.json'; saved = read_json(source_path)
        if saved.get('unit_identity') != identity:
            raise ScientificArtifactError('Clean fitted-state seal differs')
        prior = saved['fit']; states = prior['states']
        top_name = 'top_q_raw' if 'top_q_raw' in states else 'top_q'
        if top_name not in states or 'full_sgd' not in states or 'support_ea' not in states:
            raise ScientificArtifactError('Clean raw intervention states missing')
        # A nonfallback top state must be the genuine raw update, not a
        # reference-scaled candidate silently relabelled as raw.
        if states[top_name].get('layer') not in ('raw', 'anchor_fallback'):
            return None
        chosen = {}
        for name, source_name in [('support_ea', 'support_ea'), ('full_sgd', 'full_sgd'), ('top_q', top_name)]:
            chosen[name] = dict(copy.deepcopy(states[source_name]), method=name,
                                layer='anchor' if name == 'support_ea' else 'shift_raw_control')
        diagnostics = copy.deepcopy(prior.get('diagnostics', {}))
        diagnostics['reused_clean_support_fit'] = _descriptor(source_path, _output(plan))
        diagnostics['query_metrics_used_for_reuse'] = False
        return dict(states=chosen, diagnostics=diagnostics, tau_observation=prior.get('tau_observation'))
    return None


def _metric_rows(task, session, budget, fitted, labels, prediction_values, state_keys,
                 model, support, algorithm, query_identity, intervention,
                 clean_model=None, clean_support=None, frozen_model=None):
    states = fitted['states']; diagnostics = fitted.get('diagnostics', {})
    anchor_name = next((key for key in state_keys if states[key].get('method') == 'support_ea'), None)
    if anchor_name is None:
        raise ScientificArtifactError('Every paired experiment needs its correct EA anchor')
    anchor = prediction_values[state_keys.index(anchor_name)]
    weak_rms = diagnostics.get('weak_deleted_reference_rms')
    if weak_rms is None:
        weak_rms = diagnostics.get('steps', {}).get('100', {}).get('real_weak_deletion_reference_rms')
    initial = diagnostics.get('initial_spectrum', {})
    rows = []
    for index, key in enumerate(state_keys):
        state = states[key]; method, draw = _base_method(state.get('method', key))
        scoring_model = (clean_model if method == 'oracle_clean_ea' else
                         frozen_model if method == 'frozen' else model)
        scoring_support = clean_support if method == 'oracle_clean_ea' else support
        step_weak = diagnostics.get('steps', {}).get(str(state.get('steps')), {}).get(
            'real_weak_deletion_reference_rms', weak_rms)
        score = session_metrics(labels, prediction_values[index], anchor)
        row = dict(task_key=task['key'], task_kind=task['kind'], role=task['role'],
            participant_id=str(session.participant_id), participant=str(session.participant_id),
            session_id=str(session.session_id), session=str(session.session_id),
            seed=int(task['seed']), budget=int(budget),
            method=(method+'__'+intervention if task['kind'] == 'shift_block' else method),
            base_method=method, state_key=key,
            layer=_metric_layer(state), state_layer=state.get('layer', 'reference_rms'),
            steps=int(state.get('steps', 100)),
            draw_index=draw, intervention=intervention, query_identity=query_identity,
            source_identity=digest(getattr(scoring_model, 'identity', {})),
            n_valid_support=len(support.y), n_acquired_support=len(support.acquired_event_uids),
            support_BCE=_support_score(algorithm, scoring_model, state, scoring_support),
            reference_RMS=state.get('reference_rms'),
            target_reference_rms=state.get('target_reference_rms'),
            match_absolute_error=state.get('match_absolute_error'),
            weak_deleted_reference_RMS=step_weak,
            tau_observation=fitted.get('tau_observation'),
            tau_used=diagnostics.get('tau_used'),
            q=diagnostics.get('q', initial.get('q')),
            positive_rank=diagnostics.get('positive_rank', initial.get('positive_rank')),
            parameter_norm=state.get('parameter_norm'), scale=state.get('scale'),
            fallback_reason=state.get('fallback_reason'), fit_status=state.get('fit_status'),
            source_head_frozen=state.get('head_mode', 'frozen') == 'frozen',
            source_head_parameters_frozen=True, delta_head_mode=state.get('head_mode', 'frozen'),
            inference_scope=('synthetic_intervention_on_real_development_only' if task['kind'] == 'shift_block'
                             else 'development' if task['role'] in ('pilot', 'meta_validation')
                             else 'exploratory_reused_cohort'), **score)
        rows.append(_jsonable(row))
    return rows


def _paired_unit(plan, task, provider, reference, session, budget, folder, algorithm,
                 progress, *, intervention='clean', generator=None, amplitude=None, shift_scope=None):
    output = _output(plan); unit = _unit_folder(folder, session, budget, intervention)
    support = provider.support(session, budget)
    original_support = support
    if generator is not None:
        support = _replace_x(support, provider.perturb(task['seed'], support.x, generator, amplitude))
    config = _config(plan, task, session, budget, intervention)
    model = provider.episode_model(task['seed'], support)
    identity = dict(task_identity=digest(task), recipe_sha256=plan.get('recipe_sha256'),
        session_identity=session.identity, budget=int(budget),
        support_event_uids=list(support.event_uids), acquired_event_uids=list(support.acquired_event_uids),
        source_identity=_jsonable(getattr(model, 'identity', {})),
        reference_identity=_jsonable(reference['identity']), config_sha256=digest(config),
        intervention=intervention, fixed_tau=_tau(plan, budget))
    existing = _unit_receipt(unit, identity, output)
    if existing is not None:
        return existing
    _notice(folder, task, progress, stage='support_fit', session=str(session.session_id),
            budget=budget, intervention=intervention, status='FITTING')
    states_path = unit/'states.json'
    if states_path.exists():
        saved = read_json(states_path)
        if saved.get('unit_identity') != identity or not isinstance(saved.get('fit'), dict):
            raise ScientificArtifactError('Partial fitted-state identity differs; preserving files')
        fitted = saved['fit']
    else:
        raw_cache, checkpoint = _raw_cache(unit)
        kwargs = dict(budget=int(budget), seed=int(task['seed']), config=config,
                      tau=identity['fixed_tau'], raw_cache=raw_cache, checkpoint=checkpoint,
                      progress=lambda **fields: _notice(folder, task, progress, **dict(
                          dict(session=str(session.session_id), budget=budget, intervention=intervention), **fields)))
        fitted = (_reuse_clean_shift_fit(plan, task, session, budget, support, model, reference)
                  if task['kind'] == 'shift_block' and intervention == 'clean' else None)
        if fitted is None:
            fitter = (algorithm.fit_head_controls if task['kind'] == 'head_block' else
                      algorithm.fit_shift_controls if task['kind'] == 'shift_block' else algorithm.fit_episode)
            fitted = fitter(model, support.x, support.y, reference['x'], **kwargs)
        if not isinstance(fitted, dict) or not fitted.get('states'):
            raise ScientificArtifactError('Scientific fitter returned no actual prediction states')
        atomic_json(states_path, {'unit_identity': identity, 'fit': _jsonable(fitted)})
    frozen_model = provider.model(task['seed'])
    if 'frozen' not in fitted['states']:
        fitted['states']['frozen'] = dict(theta=[0.]*int(frozen_model.dimension),
            method='frozen', layer='anchor', steps=0, fit_status='FROZEN_SOURCE_REFERENCE',
            fallback_reason=None, reference_rms=0.)
        atomic_json(states_path, {'unit_identity': identity, 'fit': _jsonable(fitted)})
    clean_model = None
    if task['kind'] == 'shift_block':
        clean_model = provider.episode_model(task['seed'], original_support)
        if 'oracle_clean_ea' not in fitted['states']:
            # This state is fixed before future inputs/labels are opened.
            fitted['states']['oracle_clean_ea'] = dict(theta=[0.]*int(clean_model.dimension),
                method='oracle_clean_ea', layer='shift_oracle', steps=0,
                fit_status='KNOWN_INVERSE_ORACLE', fallback_reason=None, reference_rms=0.)
            atomic_json(states_path, {'unit_identity': identity, 'fit': _jsonable(fitted)})
    query = provider.query_inputs(session)  # Never supplied to the fitting function.
    state_keys = sorted(fitted['states'])
    prediction_path = unit/'predictions.npz'; prediction_seal = unit/'prediction-seal.json'
    if prediction_path.exists() and prediction_seal.exists():
        seal = read_json(prediction_seal)
        if seal.get('unit_identity') != identity or seal.get('states_sha256') != sha_file(states_path):
            raise ScientificArtifactError('Existing prediction seal differs from fitted state')
        _verify_files(output, [seal['prediction']])
        with np.load(prediction_path, allow_pickle=False) as saved:
            prediction_values = np.array(saved['all_probabilities'])
            if list(saved['state_keys'].tolist()) != state_keys:
                raise ScientificArtifactError('Sealed prediction method coverage changed')
    else:
        _notice(folder, task, progress, stage='fixed_query_prediction', session=str(session.session_id),
                budget=budget, intervention=intervention, status='PREDICTING')
        query_x = (provider.perturb(task['seed'], query.x, generator, amplitude)
                   if generator is not None and shift_scope == 'persistent' else query.x)
        values = [probabilities(algorithm.predict(
                    clean_model if fitted['states'][key].get('method') == 'oracle_clean_ea' else
                    frozen_model if fitted['states'][key].get('method') == 'frozen' else model,
                    fitted['states'][key],
                    query.x if fitted['states'][key].get('method') == 'oracle_clean_ea' else query_x),
                    len(query.event_uids)) for key in state_keys]
        prediction_values = np.stack(values)
        anchor_index = next((i for i, key in enumerate(state_keys)
                             if fitted['states'][key].get('method') == 'support_ea'), None)
        if anchor_index is None:
            raise ScientificArtifactError('Correct EA prediction missing before opening query labels')
        _npz(prediction_path, probabilities=prediction_values[anchor_index],
             all_probabilities=prediction_values, state_keys=np.asarray(state_keys, dtype=np.str_),
             event_uids=np.asarray(query.event_uids, dtype=np.str_),
             session_identity=np.asarray(query.identity), unit_identity_sha256=np.asarray(digest(identity)),
             states_sha256=np.asarray(sha_file(states_path)))
        seal = dict(unit_identity=identity, states_sha256=sha_file(states_path),
                    prediction=_descriptor(prediction_path, output))
        atomic_json(prediction_seal, seal)
    # Validate every method coverage before labels, including resumed bundles.
    if prediction_values.shape != (len(state_keys), len(query.event_uids), 2):
        raise ScientificArtifactError('Saved future prediction coverage differs')
    for value in prediction_values:
        probabilities(value, len(query.event_uids))
    labels = provider.evaluate_labels(session, prediction_path,
        {'sha256': seal['prediction']['sha256'], 'session_identity': query.identity})
    _npz(unit/'evaluation-labels.npz', labels=np.asarray(labels, dtype=np.int64),
         event_uids=np.asarray(query.event_uids, dtype=np.str_))
    rows = _metric_rows(task, session, budget, fitted, labels, prediction_values,
                        state_keys, model, support, algorithm, query.identity, intervention,
                        clean_model=clean_model, clean_support=original_support, frozen_model=frozen_model)
    _csv(unit/'metrics.csv', rows)
    paths = [states_path, prediction_path, prediction_seal, unit/'evaluation-labels.npz', unit/'metrics.csv']
    paths += sorted((unit/'raw-fits').glob('*.json'))
    value = dict(status='COMPLETE', unit_identity=identity, n_rows=len(rows),
                 files=[_descriptor(path, output) for path in paths],
                 metrics_csv=_descriptor(unit/'metrics.csv', output), completed_utc=utc())
    atomic_json(unit/'receipt.json', value)
    return value


def _calibration(plan, task, provider, folder, progress, stop, algorithm):
    reference = provider.reference(int(task['seed']))
    reference_record = dict(identity=_jsonable(reference['identity']),
        event_uids=list(reference['event_uids']), reference_transform=_jsonable(reference.get('reference_transform')),
        source_identity=_jsonable(getattr(provider.model(task['seed']), 'identity', {})),
        target_query_inputs_read=False, target_query_labels_read=False)
    atomic_json(folder/'reference-identity.json', reference_record)
    budgets = plan.get('config', {}).get('data', {}).get('acquired_budgets', [4, 8, 12, 16, 20])
    values = {str(budget): [] for budget in budgets}; paths = [folder/'reference-identity.json']; units = 0
    people = provider.participants('pilot')
    if not people:
        raise ValueError('Frozen pilot participant role is empty')
    for person in people:
        sessions = provider.sessions('pilot', person)
        if not sessions:
            raise ValueError('Pilot participant has no registered evaluable sessions')
        for session in sessions:
            for budget in budgets:
                _stop(stop); unit = _unit_folder(folder, session, budget, 'support-only-scale')
                support = provider.support(session, budget); model = provider.episode_model(task['seed'], support)
                config = _config(plan, dict(task, role='pilot', participant=person), session, budget, 'tau-calibration')
                identity = dict(task_identity=digest(task), recipe_sha256=plan.get('recipe_sha256'),
                    session_identity=session.identity, budget=budget,
                    source_identity=_jsonable(getattr(model, 'identity', {})),
                    reference_identity=_jsonable(reference['identity']), config_sha256=digest(config),
                    support_uids=list(support.event_uids), acquired_uids=list(support.acquired_event_uids))
                existing = _unit_receipt(unit, identity, _output(plan))
                if existing is None:
                    raw_cache, checkpoint = _raw_cache(unit)
                    result = algorithm.calibrate_tau(model, support.x, support.y, reference['x'],
                        budget=budget, seed=int(task['seed']), config=config, raw_cache=raw_cache,
                        checkpoint=checkpoint, progress=lambda **fields: _notice(folder, task, progress, **dict(
                            dict(participant=str(person), session=str(session.session_id), budget=budget), **fields)))
                    rho = float(result['tau_observation'])
                    if not math.isfinite(rho) or rho < 0:
                        raise ScientificArtifactError('Support-only reference scale is not finite')
                    atomic_json(unit/'calibration.json', dict(unit_identity=identity, fit=_jsonable(result)))
                    unitpaths = [unit/'calibration.json']+sorted((unit/'raw-fits').glob('*.json'))
                    existing = dict(status='COMPLETE', unit_identity=identity, tau_observation=rho,
                        files=[_descriptor(path, _output(plan)) for path in unitpaths], completed_utc=utc())
                    atomic_json(unit/'receipt.json', existing)
                values[str(budget)].append(float(existing['tau_observation'])); units += 1
                paths.extend(_output(plan)/entry['path'] for entry in existing['files'])
                paths.append(unit/'receipt.json')
                _notice(folder, task, progress, stage='support_only_tau_calibration', budget=budget,
                    participant=str(person), session=str(session.session_id), completed_units=units,
                    query_inputs_read=False, query_labels_read=False)
    atomic_json(folder/'tau-observations.json', dict(seed=int(task['seed']), tau_observations=values,
        reference_identity=reference_record, query_inputs_read=False, query_labels_read=False))
    return _seal(plan, task, dict(tau_observations=values, reference_identity=reference_record,
        completed_units=units, scientific_scope='support_only_scale_calibration',
        scientific_runtime=_jsonable(getattr(provider, 'runtime_profile', {}))),
        paths+[folder/'tau-observations.json'])


def run_task(plan, task, device='cuda:0', progress=None, stop=None, *, provider_factory=None, algorithm_module=None):
    """Execute/reuse one exact paired block; errors never become fake completion."""
    folder = _folder(plan, task); folder.mkdir(parents=True, exist_ok=True)
    with lock(folder/'science.lock', blocking=False):
        previous = _existing_result(plan, task)
        if previous is not None:
            return previous
        _stop(stop)
        kind = task.get('kind')
        if kind in CONTROL_KINDS:
            from .reports import run_control_task
            result = run_control_task(plan, task, progress=progress, stop=stop)
            if not isinstance(result, dict) or result.get('status') != 'COMPLETE':
                raise ScientificArtifactError('Control task did not actually complete')
            return _seal(plan, task, result)
        if kind not in ('calibration', 'paired_block', 'head_block', 'shift_block'):
            raise ValueError('Unknown registered scientific task kind: '+str(kind))
        if kind in ('head_block', 'shift_block') and task.get('role') != 'pilot':
            raise ValueError('Secondary head/known-shift interventions are development-only')
        if algorithm_module is None:
            from . import algorithm as algorithm_module
        algorithm = algorithm_module; provider = _provider(plan, device, provider_factory)
        if kind == 'calibration':
            return _calibration(plan, task, provider, folder, progress, stop, algorithm)
        seed = int(task['seed']); reference = provider.reference(seed)
        sessions = provider.sessions(task['role'], task['participant'])
        if not sessions:
            raise ValueError('Frozen participant has no eligible fixed-query sessions')
        budgets = plan.get('config', {}).get('data', {}).get('acquired_budgets', [4, 8, 12, 16, 20])
        interventions = [('clean', None, None, None)]
        if kind == 'shift_block':
            interventions += [('shift-%s-g%d-a%s'%(scope, generator, str(amplitude).replace('.', 'p')),
                               generator, amplitude, scope)
                             for scope in ('persistent', 'support_only')
                             for generator in (0, 1) for amplitude in (.05, .15)]
        metrics = []; paths = []; mechanism_episodes = []; units = 0
        for session_index, session in enumerate(sessions):
            for budget in budgets:
                for name, generator, amplitude, scope in interventions:
                    _stop(stop)
                    receipt = _paired_unit(plan, task, provider, reference, session, int(budget),
                        folder, algorithm, progress, intervention=name, generator=generator,
                        amplitude=amplitude, shift_scope=scope)
                    unit_rows = _rows(_output(plan)/receipt['metrics_csv']['path'])
                    metrics.extend(unit_rows)
                    paths.extend(_output(plan)/item['path'] for item in receipt['files'])
                    unit_folder = _unit_folder(folder, session, budget, name)
                    states_path = unit_folder/'states.json'
                    saved = read_json(states_path)
                    # Four-dimensional spectra, response targets and measured
                    # support losses are small enough to archive in channel A.
                    # Dense trial predictions and checkpoints remain in B.
                    mechanism_episodes.append(dict(participant=str(session.participant_id),
                        session=str(session.session_id), session_identity=session.identity,
                        budget=int(budget), intervention=name,
                        unit_identity_sha256=digest(receipt['unit_identity']),
                        states=_descriptor(states_path, _output(plan)),
                        diagnostics=saved['fit'].get('diagnostics', {}),
                        support_losses=[{key: row.get(key) for key in (
                            'method', 'base_method', 'layer', 'steps', 'draw_index', 'support_BCE',
                            'reference_RMS', 'target_reference_rms', 'match_absolute_error',
                            'parameter_norm', 'scale', 'fallback_reason', 'fit_status')}
                            for row in unit_rows]))
                    paths.append(unit_folder/'receipt.json'); units += 1
                    _notice(folder, task, progress, stage='paired_unit_complete', status='RUNNING',
                        completed_units=units, session=session_index+1, sessions_total=len(sessions),
                        budget=budget, intervention=name, numerical_progress=True)
        expected = len(sessions)*len(budgets)*len(interventions)
        if units != expected or not metrics:
            raise ScientificArtifactError('Incomplete paired scientific block coverage')
        _csv(folder/'metrics.csv', metrics)
        atomic_json(folder/'mechanism_diagnostics.json', dict(
            schema_version='paper3-m12-paired-mechanism-diagnostics-v1', task_key=task['key'],
            task_kind=kind, role=task['role'], source_seed=seed,
            source_states_not_selected_by_query_metrics=True, episodes=mechanism_episodes))
        from .reports import summarize_task_metrics
        metrics_descriptor = _descriptor(folder/'metrics.csv', _output(plan))
        summary = summarize_task_metrics(plan, task, metrics_descriptor)
        atomic_json(folder/'participant_summary.json', summary)
        result = dict(metrics_csv=_descriptor(folder/'metrics.csv', _output(plan)), n_rows=len(metrics),
            participant_summary=_descriptor(folder/'participant_summary.json', _output(plan)),
            mechanism_diagnostics=_descriptor(folder/'mechanism_diagnostics.json', _output(plan)),
            completed_units=units, expected_units=expected, n_sessions=len(sessions), budgets=list(budgets),
            interventions=[item[0] for item in interventions], source_seed=seed,
            scientific_runtime=_jsonable(getattr(provider, 'runtime_profile', {})),
            participant=str(task['participant']), role=task['role'],
            primary_eligible=kind == 'paired_block',
            scientific_scope=('development' if task['role'] in ('pilot', 'meta_validation')
                              else 'exploratory_reused_cohort'))
        return _seal(plan, task, result, paths+[folder/'metrics.csv', folder/'participant_summary.json',
                                               folder/'mechanism_diagnostics.json'])
