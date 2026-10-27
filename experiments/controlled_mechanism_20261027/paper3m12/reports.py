"""Participant-paired M12 reports and prespecified development decisions.

Reporting is independent of fitting. Only scientifically completed task receipts
and their declared, sealed metric files are read. Source seeds, sessions and
random subspace draws never become additional independent participants.
"""
from __future__ import annotations

import csv
from collections import OrderedDict, defaultdict
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import random
import re
from statistics import NormalDist, mean, median, stdev

from .common import atomic_bytes, atomic_json, digest, read_json, sha_file, utc, verify_result, lock

PRIMARY = (
    ('P1', 'top_q', 'random_q', 'BA', 'positive'),
    ('P2', 'top_q', 'random_q', 'balanced_NLL', 'negative'),
    ('P3', 'drop_weak', 'drop_random', 'BA', 'positive'),
    ('P4', 'drop_weak', 'drop_random', 'balanced_NLL', 'negative'),
    ('P5', 'top_q', 'scalar_shrink_full', 'BA', 'positive'),
    ('P6', 'top_q', 'support_ea', 'BA', 'positive'),
)
METRICS = ('BA', 'balanced_NLL', 'Brier', 'support_BCE', 'reference_RMS',
           'weak_deleted_reference_RMS', 'corrected_vs_EA', 'introduced_vs_EA',
           'match_absolute_error', 'target_reference_rms')
_FILE_CACHE = OrderedDict()
_CACHE_LIMIT = 1024


def _config(plan):
    return plan.get('config', plan.get('experiment_config', {}))


def _seeds(plan):
    return tuple(int(x) for x in _config(plan).get('model', {}).get(
        'source_seeds', plan.get('source_seeds', [20261018, 20261019, 20261020])))


def _role(task):
    return str(task.get('role', task.get('cohort', 'unspecified')))


def _person(task):
    return str(task.get('participant', task.get('participant_id', '')))


def _paired_tasks(plan, role=None):
    return [task for task in plan.get('tasks', [])
            if task.get('kind') in ('paired_block', 'evaluate_block')
            and (role is None or _role(task) == role)]


def completed_results(plan, *, kinds=None):
    """Yield one small receipt at a time; no campaign-sized JSON allocation."""
    for task in plan.get('tasks', []):
        if kinds is not None and task.get('kind') not in kinds:
            continue
        result = verify_result(plan, task)
        if result is not None:
            yield task, result


def _descriptors(result):
    values = result.get('files', result.get('artifacts', []))
    if isinstance(values, dict):
        return [dict(item, path=item.get('path', name)) for name, item in values.items()
                if isinstance(item, dict)]
    return [item for item in values if isinstance(item, dict)]


def declared_path(plan, item, *, verify=True):
    """Resolve only a declared scientific-output artifact and cache its SHA proof."""
    if not isinstance(item, dict) or not isinstance(item.get('path'), str):
        raise ValueError('Scientific artifact descriptor required')
    root = Path(plan['output']).resolve()
    candidate = Path(item['path'])
    path = candidate if candidate.is_absolute() else root/candidate
    if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError('Scientific artifact is not a regular file below output')
    stat = path.stat()
    signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    expected = (item.get('bytes'), item.get('sha256'))
    if verify:
        if type(expected[0]) is not int or expected[0] != stat.st_size:
            raise ValueError('Sealed scientific artifact size differs')
        if not isinstance(expected[1], str) or not re.fullmatch(r'[0-9a-f]{64}', expected[1]):
            raise ValueError('Sealed scientific artifact SHA256 required')
        key = (str(path), signature, expected)
        if key not in _FILE_CACHE:
            if sha_file(path) != expected[1]:
                raise ValueError('Sealed scientific artifact SHA256 differs')
            _FILE_CACHE[key] = True
            while len(_FILE_CACHE) > _CACHE_LIMIT:
                _FILE_CACHE.popitem(last=False)
        else:
            _FILE_CACHE.move_to_end(key)
    return path


def metric_descriptor(result):
    explicit = result.get('metrics_csv')
    if isinstance(explicit, dict):
        return explicit
    descriptors = _descriptors(result)
    if isinstance(explicit, str):
        for item in descriptors:
            if item.get('path') == explicit:
                return item
        raise ValueError('Metrics CSV was not sealed in scientific result')
    matches = [item for item in descriptors
               if Path(item.get('path', '')).name in ('metrics.csv', 'session_metrics.csv')]
    if len(matches) > 1:
        raise ValueError('Scientific task has ambiguous aggregate metrics CSV')
    return matches[0] if matches else None


def _finite(value, required=False):
    if value in (None, '', 'None', 'null'):
        if required:
            raise ValueError('Required scientific metric is missing')
        return None
    value = float(value)
    if not math.isfinite(value):
        if required:
            raise ValueError('Required scientific metric is non-finite')
        return None
    return value


def _normalized(row, task):
    method = str(row.get('method', row.get('state_key', '')))
    draw = row.get('draw_index')
    match = re.fullmatch(r'(random_q|drop_random)_(\d+)(?P<suffix>_head_.+|__.+)?', method)
    if match:
        method, draw = match.group(1)+(match.group('suffix') or ''), int(match.group(2))
    elif method in ('random_q', 'drop_random') or method.startswith(('random_q_head_', 'drop_random_head_', 'random_q__', 'drop_random__')):
        draw = int(draw) if draw not in (None, '') else None
    else:
        draw = None
    aliases = {'support_EA': 'support_ea', 'top2': 'top_q', 'random2': 'random_q',
               'scalar-shrunk-full': 'scalar_shrink_full'}
    method = aliases.get(method, method)
    layer = str(row.get('layer', 'raw'))
    if layer in ('anchor_fallback', 'fallback'):
        layer = ('removal_reference_rms' if method.startswith('drop_') else
                 'anchor' if method == 'support_ea' else 'reference_rms')
    steps = int(row.get('steps') or 0)
    if method == 'support_ea':
        layer, steps = 'anchor', 0
    out = dict(role=str(row.get('role') or _role(task)),
               participant=str(row.get('participant') or row.get('participant_id') or _person(task)),
               session=str(row.get('session') or row.get('session_id') or ''),
               seed=int(row.get('seed') or task.get('seed')),
               budget=int(row.get('budget') or 0), method=method, layer=layer,
               steps=steps, draw_index=draw,
               query_identity=str(row.get('query_identity') or ''),
               n_query=int(row.get('n_query') or 0),
               n_valid_support=int(row.get('n_valid_support') or 0),
               fallback_reason=str(row.get('fallback_reason') or ''),
               fit_status=str(row.get('fit_status') or ''))
    if not out['participant'] or not out['session'] or not out['method']:
        raise ValueError('Metric row lacks participant/session/method identity')
    if out['role'] != _role(task) or out['participant'] != _person(task) or out['seed'] != int(task['seed']):
        raise ValueError('Metric row does not bind its completed paired task')
    for name in METRICS:
        alias = {'BA': 'ba', 'balanced_NLL': 'balanced_nll', 'Brier': 'brier'}.get(name)
        value = row.get(name, row.get(alias) if alias else None)
        out[name] = _finite(value, required=name in ('BA', 'balanced_NLL'))
    if not 0 <= out['BA'] <= 1 or out['balanced_NLL'] < 0:
        raise ValueError('Scientific metric outside its declared range')
    return out


def _average(records, key_names):
    out = {key: records[0][key] for key in key_names}
    for metric in METRICS:
        values = [record[metric] for record in records if record.get(metric) is not None]
        out[metric] = mean(values) if values else None
    return out


def aggregate_metrics(plan, role=None, *, result_records=None):
    """Draw metrics, then sessions, then source seeds; participant always unit N."""
    expected_draws = int(_config(plan).get('model', {}).get('random_draws', 5))
    groups = {}
    completed = []
    issues = []
    sealed_seed_rows = []
    weak_episode_rows = []
    matching_maxima = []
    metric_sources = []
    records_to_read = (result_records if result_records is not None else
                       completed_results(plan, kinds={'paired_block', 'evaluate_block'}))
    for task, result in records_to_read:
        if role is not None and _role(task) != role:
            continue
        item = metric_descriptor(result)
        if item is None:
            issues.append(dict(task_key=task['key'], code='sealed_metrics_missing'))
            continue
        path = declared_path(plan, item)
        completed.append(task['key'])
        metric_sources.append(dict(task_key=task['key'], role=_role(task), participant=_person(task),
                                   seed=task.get('seed'), **item))
        summary_item = result.get('participant_summary')
        if isinstance(summary_item, dict):
            metric_sources[-1]['participant_summary_sha256'] = summary_item.get('sha256')
            summary_path = declared_path(plan, summary_item)
            summary = read_json(summary_path)
            if (not isinstance(summary, dict) or summary.get('schema_version') != 'paper3-m12-participant-seed-summary-v1'
                    or summary.get('task_key') != task['key'] or summary.get('task_identity') != digest(task)
                    or summary.get('metrics_sha256') != item['sha256']):
                raise ValueError('Compact participant summary does not bind its sealed task/metrics')
            sealed_seed_rows.extend(summary.get('seed_rows', []))
            weak_episode_rows.extend(summary.get('weak_episode_rows', []))
            matching_maxima.extend(summary.get('matching_maxima', []))
            issues.extend(summary.get('issues', []))
            continue
        with path.open(newline='', encoding='utf-8') as handle:
            for original in csv.DictReader(handle):
                row = _normalized(original, task)
                key = tuple(row[name] for name in ('role', 'participant', 'seed', 'budget',
                                                   'method', 'layer', 'steps', 'session'))
                group = groups.setdefault(key, {})
                draw = row['draw_index']
                prior = group.get(draw)
                if prior is not None:
                    # Scientific state aliases can denote the identical anchor
                    # or raw full update. They cannot increase its statistical weight.
                    if any(prior.get(name) != row.get(name) for name in (*METRICS, 'query_identity', 'n_query')):
                        raise ValueError('Conflicting duplicate scientific metric identity')
                else:
                    group[draw] = row
    session_rows = []
    for key, draws in groups.items():
        records = list(draws.values())
        if records[0]['method'].startswith(('random_q', 'drop_random')) and set(draws) != set(range(expected_draws)):
            issues.append(dict(role=key[0], participant=key[1], seed=key[2], budget=key[3],
                               method=key[4], code='random_draw_coverage_incomplete'))
        identities = {(row['query_identity'], row['n_query']) for row in records}
        if len(identities) != 1:
            raise ValueError('Random draws used different future queries')
        out = _average(records, ('role', 'participant', 'seed', 'budget', 'method', 'layer', 'steps', 'session'))
        out['maximum_match_absolute_error'] = max((row['match_absolute_error'] for row in records if row['match_absolute_error'] is not None), default=None)
        out.update(query_identity=records[0]['query_identity'], n_query=records[0]['n_query'],
                   draws=len(records), n_valid_support=records[0]['n_valid_support'],
                   fallback_count=sum(bool(row['fallback_reason']) for row in records))
        session_rows.append(out)
    seed_groups = defaultdict(list)
    for row in session_rows:
        key = tuple(row[name] for name in ('role', 'participant', 'seed', 'budget', 'method', 'layer', 'steps'))
        seed_groups[key].append(row)
    seed_rows = list(sealed_seed_rows)
    for records in seed_groups.values():
        out = _average(records, ('role', 'participant', 'seed', 'budget', 'method', 'layer', 'steps'))
        out['sessions'] = len(records)
        out['session_query_signature'] = digest(sorted((row['session'], row['query_identity'], row['n_query'])
                                                       for row in records))
        seed_rows.append(out)
    participant_groups = defaultdict(list)
    for row in seed_rows:
        key = tuple(row[name] for name in ('role', 'participant', 'budget', 'method', 'layer', 'steps'))
        participant_groups[key].append(row)
    participants = []
    for records in participant_groups.values():
        out = _average(records, ('role', 'participant', 'budget', 'method', 'layer', 'steps'))
        seed_ids = sorted(row['seed'] for row in records)
        out.update(seeds=len(records), seed_ids=seed_ids,
                   complete_source_seeds=tuple(seed_ids) == tuple(sorted(_seeds(plan))),
                   session_query_signature=digest(sorted((row['seed'], row['session_query_signature'])
                                                         for row in records)),
                   session_seed_rows=sum(row['sessions'] for row in records))
        participants.append(out)
    if session_rows:
        weak_episode_rows.extend(dict(row) for row in session_rows if row['method'] == 'drop_weak'
                                 and row['budget'] == 20 and row['steps'] == 100
                                 and row['layer'] == 'removal_reference_rms')
        maxima = [row['maximum_match_absolute_error'] for row in session_rows
                  if row['layer'] in ('reference_rms', 'removal_reference_rms')
                  and row.get('maximum_match_absolute_error') is not None]
        if maxima:
            matching_maxima.append(dict(role=role, maximum_match_absolute_error=max(maxima),
                                        matched_session_conditions=len(maxima)))
    return dict(session_rows=session_rows, seed_rows=seed_rows, participant_rows=participants,
                completed_task_keys=sorted(completed), issues=issues, weak_episode_rows=weak_episode_rows,
                matching_maxima=matching_maxima, metric_sources=metric_sources)


def percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    position = fraction * (len(sorted_values)-1)
    lower = int(math.floor(position)); upper = int(math.ceil(position))
    return sorted_values[lower] + (sorted_values[upper]-sorted_values[lower]) * (position-lower)


def paired_statistics(differences, *, bootstrap_repeats=20000, seed=20261027,
                      signflip_draws=100000):
    values = [float(value) for value in differences]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError('Finite paired participant differences required')
    n = len(values); effect = mean(values)
    rng = random.Random(seed)
    bootstrap = sorted(sum(values[rng.randrange(n)] for _ in range(n))/n
                       for _ in range(bootstrap_repeats))
    observed = abs(sum(values)); tolerance = max(1e-15, observed*1e-12)
    if n <= 18:
        total = 1 << n
        extreme = sum(abs(sum(value if sign else -value for value, sign in zip(values, signs)))
                      >= observed-tolerance for signs in itertools.product((False, True), repeat=n))
        pvalue = extreme/total
        mode = 'exact_sign_flip'
    else:
        rng = random.Random(seed+1)
        extreme = 0
        for _ in range(signflip_draws):
            signed = sum(value if rng.getrandbits(1) else -value for value in values)
            extreme += abs(signed) >= observed-tolerance
        total = signflip_draws
        pvalue = (extreme+1)/(total+1)
        mode = 'monte_carlo_sign_flip_plus_one'
    return dict(effect_mean=effect, ci95=[percentile(bootstrap, .025), percentile(bootstrap, .975)],
                pvalue=pvalue, n_participants=n, participant_differences=values,
                bootstrap_repeats=bootstrap_repeats, method=mode, signflip_assignments=total,
                independent_unit='participant', ci_scope='pointwise percentile interval; not simultaneous',
                signflip_null_assumption='paired sign symmetry or exchangeability; not an automatic mean-zero guarantee')


def holm(values):
    if any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in values):
        raise ValueError('Valid original p-values required for the complete fixed family')
    result = [None]*len(values); running = 0.
    for rank, (index, value) in enumerate(sorted(enumerate(values), key=lambda row: row[1])):
        running = max(running, (len(values)-rank)*value)
        result[index] = min(1., running)
    return result


def _primary_layer(method):
    if method == 'support_ea':
        return 'anchor'
    return 'removal_reference_rms' if method.startswith('drop_') else 'reference_rms'


def _series(rows, role, method, budget=20, steps=100, layer=None):
    layer = layer or _primary_layer(method)
    return {row['participant']: row for row in rows
            if row['role'] == role and row['method'] == method and row['budget'] == budget
            and row['layer'] == layer and row['steps'] == (0 if method == 'support_ea' else steps)
            and row['complete_source_seeds']}


def _scope(role):
    if role in ('pilot', 'meta_train_pilot', 'meta_validation', 'development_validation'):
        return 'development_only'
    if role in ('confirmation', 'independent_confirmation'):
        return 'independent_confirmation_only_if_explicit_unused_data_registry'
    return 'exploratory_reused_cohort'


def _write_csv(path, rows):
    rows = list(rows)
    fields = sorted({key for row in rows for key in row})
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fields, lineterminator='\n'); writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(',', ':'))
                         if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})
    atomic_bytes(path, out.getvalue().encode())


def _describe(plan, path):
    path = Path(path)
    return dict(path=path.relative_to(Path(plan['output'])).as_posix(),
                bytes=path.stat().st_size, sha256=sha_file(path))


def _expected_people(plan, role):
    return sorted({_person(task) for task in _paired_tasks(plan, role)})


def _comparisons(plan, aggregate, role, *, bootstrap_repeats=None, signflip_draws=None):
    cfg = _config(plan).get('statistics', {})
    steps = int(_config(plan).get('optimizer', {}).get('primary_steps', 100))
    repeats = int(bootstrap_repeats or cfg.get('bootstrap_resamples', 20000))
    flips = int(signflip_draws or cfg.get('large_n_signflip_draws', 100000))
    seed = int(cfg.get('statistical_seed', 20261027))
    expected = _expected_people(plan, role)
    completed = set(aggregate['completed_task_keys'])
    all_tasks = {task['key'] for task in _paired_tasks(plan, role)}
    role_issues = [item for item in aggregate['issues'] if item.get('role', role) == role]
    rows = []
    for identifier, candidate, baseline, metric, direction in PRIMARY:
        left = _series(aggregate['participant_rows'], role, candidate, steps=steps)
        right = _series(aggregate['participant_rows'], role, baseline, steps=steps)
        people = sorted(set(left) & set(right))
        paired = [person for person in people
                  if left[person]['session_query_signature'] == right[person]['session_query_signature']]
        ready = (bool(expected) and paired == expected and all_tasks <= completed and not role_issues)
        row = dict(id=identifier, role=role, budget=20, candidate=candidate, baseline=baseline,
                   metric=metric, expected_sign=direction, n_participants=len(paired),
                   expected_participants=len(expected), participant_ids=paired,
                   status='READY' if ready else 'WAITING_FOR_COMPLETE_PAIRED_COVERAGE',
                   inference_scope=_scope(role), family_size=6, family_complete=False,
                   holm_pvalue=None, effect_mean=None, ci95=None, pvalue=None)
        if ready:
            differences = [left[person][metric]-right[person][metric] for person in paired]
            row.update(paired_statistics(differences, bootstrap_repeats=repeats,
                                         signflip_draws=flips, seed=seed))
            row['effect_pp'] = 100*row['effect_mean'] if metric == 'BA' else None
            row['ci95_pp'] = [100*x for x in row['ci95']] if metric == 'BA' else None
            row['fraction_negative_effect'] = sum(value < 0 for value in differences)/len(differences)
        rows.append(row)
    complete = all(row['status'] == 'READY' for row in rows)
    if complete:
        for row, adjusted in zip(rows, holm([row['pvalue'] for row in rows])):
            row['family_complete'] = True
            row['holm_pvalue'] = adjusted
            row['significant_holm_05'] = adjusted < .05
            row['direction_matches_hypothesis'] = (row['effect_mean'] > 0 if row['expected_sign'] == 'positive'
                                                   else row['effect_mean'] < 0)
    return rows


def _historical_context(plan):
    """Retain original historical tables, never merge them into M12 pairs."""
    source = Path(__file__).resolve().parents[1]/'evidence'
    manifest = read_json(source/'MANIFEST.json', {}) or {}
    folder = Path(plan['output'])/'reports'/'historical_context'
    artifacts = []
    names = ('R10_method_summary.csv', 'R10_comparisons.csv', 'R3_primary_comparisons.csv',
             'R7_sensitivities.csv', 'R7_sensitivity_comparisons.csv')
    for name in names:
        path = source/name
        expected = manifest.get('files', {}).get(name)
        if not path.is_file() or not isinstance(expected, dict):
            continue
        data = path.read_bytes()
        if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
            raise ValueError('Historical evidence differs from pinned original bytes')
        target = folder/name
        if not target.is_file() or sha_file(target) != expected['sha256']:
            atomic_bytes(target, data)
        artifacts.append(_describe(plan, target))
    if artifacts:
        note = dict(scope='historical_exploratory_not_current_fit', current_M12_pairing=False,
                    original_tables_retained_without_row_truncation=True,
                    warning='Historical source/query/code identities were not re-proved against new M12 outputs; never use these tables in the new paired primary family',
                    artifacts=artifacts)
        if read_json(folder/'SCOPE.json') != note:
            atomic_json(folder/'SCOPE.json', note)
        artifacts.append(_describe(plan, folder/'SCOPE.json'))
    return artifacts


def build_report(plan, role=None, *, bootstrap_repeats=None, signflip_draws=None):
    """Build complete tables when ready; incomplete families never get resized."""
    aggregate = aggregate_metrics(plan, role)
    roles = sorted({_role(task) for task in _paired_tasks(plan, role)})
    secondary = {}
    if role is None:
        for kind in ('head_block', 'shift_block'):
            secondary[kind] = aggregate_metrics(plan, result_records=completed_results(plan, kinds={kind}))
    root_folder = Path(plan['output'])/'reports'
    identity = digest(dict(schema='paper3-m12-report-snapshot-v1', recipe=plan.get('recipe_sha256'),
                           role=role, sources=aggregate['metric_sources'],
                           secondary_sources={kind:value['metric_sources'] for kind,value in secondary.items()},
                           expected_tasks=[digest(task) for task in _paired_tasks(plan, role)],
                           secondary_tasks=[digest(task) for task in plan['tasks'] if role is None and task.get('kind') in secondary],
                           bootstrap_repeats=bootstrap_repeats, signflip_draws=signflip_draws))
    folder = root_folder/'snapshots'/identity
    # This snapshot is immutable after seal; background reporting can never
    # invalidate a final_report result by overwriting a declared artifact.
    with lock(root_folder/'.snapshot.lock'):
        existing = read_json(folder/'report.json')
        if existing is not None:
            for item in existing.get('artifacts', []): declared_path(plan, item)
            existing['artifacts'].append(_describe(plan, folder/'report.json'))
            atomic_json(root_folder/'report.json', existing)
            return existing
        return _write_report_snapshot(plan, aggregate, roles, folder, root_folder, identity,
                                      bootstrap_repeats, signflip_draws, secondary)


def _write_report_snapshot(plan, aggregate, roles, folder, root_folder, identity, bootstrap_repeats, signflip_draws, secondary):
    role_tasks = _paired_tasks(plan)
    artifacts = _historical_context(plan)
    summaries = {}
    for name in roles:
        target = folder/name
        subset = {key: [row for row in aggregate[key] if row['role'] == name]
                  for key in ('session_rows', 'seed_rows', 'participant_rows')}
        comparisons = _comparisons(plan, aggregate, name,
                                  bootstrap_repeats=bootstrap_repeats, signflip_draws=signflip_draws)
        session_table = (('session_metrics.csv', subset['session_rows']) if subset['session_rows'] else
                         ('session_metrics_index.csv', [row for row in aggregate['metric_sources'] if row['role'] == name]))
        for filename, rows in [session_table,
                               ('participant_seed_metrics.csv', subset['seed_rows']),
                               ('participant_metrics.csv', subset['participant_rows']),
                               ('primary_comparisons.csv', comparisons)]:
            _write_csv(target/filename, rows); artifacts.append(_describe(plan, target/filename))
        expected = {task['key'] for task in _paired_tasks(plan, name)}
        complete = expected <= set(aggregate['completed_task_keys']) and bool(expected)
        summaries[name] = dict(expected_tasks=len(expected), complete_tasks=len(expected & set(aggregate['completed_task_keys'])),
                               comparisons=comparisons, family_complete=all(row['family_complete'] for row in comparisons),
                               status='COMPLETE' if complete and all(row['family_complete'] for row in comparisons) else 'PARTIAL',
                               inference_scope=_scope(name))
    secondary_summaries = {}
    for kind, values in secondary.items():
        expected_keys = {task['key'] for task in plan['tasks'] if task.get('kind') == kind}
        if not expected_keys:
            continue
        target = folder/'secondary'/kind
        for name, rows in [('session_metrics_index.csv', values['metric_sources']),
                           ('participant_seed_metrics.csv', values['seed_rows']),
                           ('participant_metrics.csv', values['participant_rows'])]:
            _write_csv(target/name, rows);artifacts.append(_describe(plan, target/name))
        done = expected_keys & set(values['completed_task_keys'])
        secondary_summaries[kind] = dict(expected_tasks=len(expected_keys), complete_tasks=len(done),
            status='COMPLETE' if done == expected_keys and not values['issues'] else 'PARTIAL',
            inference_scope='secondary_development_mechanism_only', primary_family_included=False,
            random_draws_averaged_as_metrics=True, independent_unit='participant',
            intervention_identity_preserved_in_method_names=True, bootstrap_or_multiple_test_claims=False,
            issues=values['issues'])
        atomic_json(target/'SCOPE.json', secondary_summaries[kind]);artifacts.append(_describe(plan,target/'SCOPE.json'))
    expected_tasks = [task for task in role_tasks if _role(task) in roles]
    report = dict(schema_version='paper3-m12-report-v1', round=12, status=(
        'COMPLETE' if summaries and all(row['status'] == 'COMPLETE' for row in summaries.values())
        and all(row['status'] == 'COMPLETE' for row in secondary_summaries.values()) else 'PARTIAL'),
        recipe_sha256=plan.get('recipe_sha256'), snapshot_sha256=identity, updated_utc=utc(), roles=summaries, secondary=secondary_summaries,
        total_paired_tasks=len(expected_tasks), complete_paired_tasks=len(aggregate['completed_task_keys']),
        complete_task_keys=aggregate['completed_task_keys'], issues=aggregate['issues'],
        artifacts=artifacts, scientific_completion_claimed=False,
        independent_confirmation_available=bool(plan.get('independent_confirmation_registry')),
        independent_unit='participant', training_controlled=False,
        full_session_CSVs_retained_per_task=True)
    atomic_json(folder/'report.json', report)
    report['artifacts'].append(_describe(plan, folder/'report.json'))
    atomic_json(root_folder/'report.json', report)
    return report


def _result_by_key(plan):
    return {task['key']: (task, result) for task, result in completed_results(plan)}


def _control_descriptor(plan, path):
    return [_describe(plan, path)]


def _dependencies(task):
    return list(task.get('dependencies', task.get('depends_on', [])))


def _complete_role(plan, role):
    expected = _paired_tasks(plan, role)
    found = {task['key'] for task, result in completed_results(plan, kinds={'paired_block', 'evaluate_block'})
             if _role(task) == role}
    if not expected or {task['key'] for task in expected} != set(found):
        raise RuntimeError('Prespecified role has incomplete scientific task coverage')
    if set(int(task['seed']) for task in expected) != set(_seeds(plan)):
        raise ValueError('Prespecified role lacks all frozen source seeds')
    return expected


def _point_criteria(plan, role, aggregate, comparisons):
    if not all(row['family_complete'] for row in comparisons):
        raise RuntimeError('Development decision requires complete paired primary metrics')
    points = {row['id']: row['effect_mean'] for row in comparisons}
    full = _series(aggregate['participant_rows'], role, 'full_sgd', layer='raw')
    anchor = _series(aggregate['participant_rows'], role, 'support_ea')
    people = _expected_people(plan, role)
    if sorted(full) != people or sorted(anchor) != people:
        raise RuntimeError('Development decision lacks unscaled full-SGD/anchor coverage')
    support = [full[person]['support_BCE']-anchor[person]['support_BCE'] for person in people
               if full[person]['support_BCE'] is not None and anchor[person]['support_BCE'] is not None]
    support_evidence_complete = len(support) == len(people)
    mean_support = mean(support) if support_evidence_complete else None
    future = mean(full[person]['balanced_NLL']-anchor[person]['balanced_NLL'] for person in people)
    episode_values = defaultdict(list)
    for row in aggregate.get('weak_episode_rows', aggregate['session_rows']):
        if row['role'] == role and row['method'] == 'drop_weak' and row['budget'] == 20 and row['steps'] == 100:
            value = row.get('weak_deleted_reference_RMS')
            if value is not None and row['layer'] == 'removal_reference_rms':
                episode_values[(row['participant'], row['session'])].append(value)
    expected_sources = len(_seeds(plan))
    weak_evidence_complete = bool(episode_values) and all(len(values) == expected_sources for values in episode_values.values())
    meaningful = mean(mean(values) > 1e-4 for values in episode_values.values()) if weak_evidence_complete else None
    criteria = dict(
        development_support_loss_evidence_complete=support_evidence_complete,
        actual_weak_deletion_evidence_complete=weak_evidence_complete,
        support_fit_improves_and_future_NLL_worsens=mean_support is not None and mean_support < 0 and future > 0,
        actual_weak_deletion_has_reference_effect_in_half_episodes=meaningful is not None and meaningful >= .5,
        top_vs_random_BA_positive=points['P1'] > 0,
        top_vs_random_NLL_nonpositive=points['P2'] <= 0,
        drop_weak_vs_random_BA_positive=points['P3'] > 0,
        drop_weak_vs_random_NLL_nonpositive=points['P4'] <= 0,
        top_vs_scalar_shrink_BA_nonnegative=points['P5'] >= 0)
    return criteria, dict(mean_support_BCE_delta_full_minus_EA=mean_support,
                          support_loss_participants_available=len(support),
                          support_loss_participants_expected=len(people),
                          insufficient_evidence=not support_evidence_complete or not weak_evidence_complete,
                          mean_future_NLL_delta_full_minus_EA=future,
                          meaningful_weak_deletion_fraction=meaningful,
                          meaningful_weak_deletion_unit='participant-session, mean across source seeds',
                          primary_point_effects=points)


def _numeric_stability(plan, aggregate, role):
    tolerance = float(_config(plan).get('matching', {}).get('absolute_match_tolerance', 1e-5))
    maxima = [row for row in aggregate.get('matching_maxima', []) if row.get('role') in (None, role)]
    errors = [row['maximum_match_absolute_error'] for row in maxima
              if row.get('maximum_match_absolute_error') is not None]
    return bool(errors) and all(value <= tolerance+1e-12 for value in errors), dict(
        matched_session_conditions=sum(row.get('matched_session_conditions', 0) for row in maxima),
        maximum_match_error=max(errors) if errors else None,
        absolute_match_tolerance=tolerance)


def summarize_task_metrics(plan, task, metrics_csv):
    """Compact once after a block; future reports avoid re-parsing its full CSV.

    The worker calls this only after predictions and metrics are sealed. The
    summary retains equal-session/equal-draw estimates and a tiny episode gate
    index; the original full session CSV remains a Channel-A artifact.
    """
    aggregate = aggregate_metrics(plan, _role(task), result_records=[(task, dict(metrics_csv=metrics_csv))])
    return dict(schema_version='paper3-m12-participant-seed-summary-v1',
                task_key=task['key'], task_identity=digest(task), metrics_sha256=metrics_csv['sha256'],
                seed_rows=aggregate['seed_rows'], weak_episode_rows=aggregate['weak_episode_rows'],
                matching_maxima=aggregate['matching_maxima'], issues=aggregate['issues'],
                random_draws_averaged_as_metrics=True, independent_unit='participant',
                original_session_metrics_retained=True)


def run_control_task(plan, task, *, progress=None, stop=None):
    """Return an actual control result; worker seals the returned artifact last."""
    kind = task['kind']; output = Path(plan['output'])
    if stop is not None and stop.is_set():
        raise InterruptedError('Graceful stop before independent control unit')
    if kind == 'freeze_tau':
        found = _result_by_key(plan)
        dependencies = _dependencies(task)
        calibrations = [(planned, result) for planned, result in found.values()
                        if planned['key'] in dependencies and planned.get('kind') == 'calibration']
        if len(calibrations) != len(_seeds(plan)) or set(int(row[0]['seed']) for row in calibrations) != set(_seeds(plan)):
            raise RuntimeError('Tau freeze requires all frozen support-only calibration seeds')
        budgets = _config(plan).get('data', {}).get('acquired_budgets', [4, 8, 12, 16, 20])
        values = defaultdict(list)
        for planned, result in calibrations:
            if result.get('query_labels_used') is True or result.get('future_query_labels_opened', 0):
                raise ValueError('Tau calibration may not open future labels')
            for budget in budgets:
                batch = result.get('tau_observations', {}).get(str(budget))
                if not isinstance(batch, list) or not batch:
                    raise RuntimeError('Tau calibration lacks a registered budget')
                for value in batch:
                    value = float(value)
                    if not math.isfinite(value) or value < 0:
                        raise ValueError('Tau observations must be finite and nonnegative')
                    values[str(budget)].append(value)
        body = dict(schema_version='paper3-m12-frozen-tau-v1', recipe_sha256=plan.get('recipe_sha256'),
                    tau_by_budget={str(budget): median(values[str(budget)]) for budget in budgets},
                    observation_counts={str(budget): len(values[str(budget)]) for budget in budgets},
                    calibration_task_keys=sorted(planned['key'] for planned, _ in calibrations),
                    calibration_input_sha256=digest([(planned['key'], result.get('tau_observations'))
                                                    for planned, result in calibrations]),
                    support_only=True, frozen_before_pilot_query_evaluation=True,
                    target_query_labels_used=False, created_utc=utc())
        path = output/'frozen_tau.json'; previous = read_json(path)
        if previous is not None:
            if any(previous.get(key) != body.get(key) for key in ('recipe_sha256', 'tau_by_budget', 'calibration_input_sha256')):
                raise ValueError('Existing frozen tau cannot be revised after calibration')
            body = previous
        else:
            if any(_role(row) in ('pilot', 'meta_train_pilot') for row, _ in completed_results(
                    plan, kinds={'paired_block', 'evaluate_block'})):
                raise ValueError('Cannot newly freeze tau after pilot evaluation results exist')
            atomic_json(path, body)
        return dict(status='COMPLETE', task=task, tau_by_budget=body['tau_by_budget'],
                    tau_sha256=sha_file(path), artifacts=_control_descriptor(plan, path),
                    target_query_labels_used=False)
    if kind in ('pilot_decision', 'development_freeze'):
        role = task.get('role', 'pilot' if kind == 'pilot_decision' else 'meta_validation')
        _complete_role(plan, role)
        report = build_report(plan, role)
        aggregate = aggregate_metrics(plan, role)
        comparisons = report['roles'][role]['comparisons']
        criteria, evidence = _point_criteria(plan, role, aggregate, comparisons)
        stable, stability = _numeric_stability(plan, aggregate, role)
        criteria['numeric_reference_matching_stable'] = stable
        decision = 'GO' if all(criteria.values()) else 'STOP'
        body = dict(schema_version='paper3-m12-development-decision-v1', task_key=task['key'],
                    role=role, decision=decision,
                    decision_status='STOP_INSUFFICIENT_EVIDENCE' if evidence.get('insufficient_evidence') else decision,
                    criteria=criteria, evidence=evidence,
                    stability=stability, updated_utc=utc(), recipe_sha256=plan.get('recipe_sha256'),
                    statistical_significance_claimed=False, future_risk_guarantee=False,
                    mechanism_family_revised=False, independent_confirmation_available=False)
        if kind == 'development_freeze':
            tau = read_json(output/'frozen_tau.json')
            if not isinstance(tau, dict):
                raise RuntimeError('Development freeze requires the already frozen support-only tau')
            p1 = next(row for row in comparisons if row['id'] == 'P1')
            sd = stdev(p1['participant_differences']) if len(p1['participant_differences']) > 1 else None
            n = (math.ceil(((NormalDist().inv_cdf(1-.05/12)+NormalDist().inv_cdf(.8))*sd/.015)**2)
                 if sd is not None else None)
            body.update(configuration_sha256=digest(_config(plan)), frozen_tau_sha256=sha_file(output/'frozen_tau.json'),
                        input_registry=plan.get('input_registry', plan.get('study_registry')),
                        method_selection='Predeclared methods retained; no target-result search',
                        observed_P1_participant_difference_sd=sd,
                        prospective_BA_example_recommended_complete_n=n,
                        planned_new_evaluable_participants_unchanged=90,
                        independent_data_status='DECLARED_NOT_AVAILABLE')
        path = output/('pilot-decision.json' if kind == 'pilot_decision' else 'development-freeze.json')
        atomic_json(path, body)
        return dict(status='COMPLETE', task=task, decision=decision, criteria=criteria,
                    artifacts=[_describe(plan, path)], report_status=report['status'])
    if kind == 'final_report':
        report = build_report(plan)
        return dict(status='COMPLETE', task=task, report_status=report['status'],
                    science_complete=report['status'] == 'COMPLETE', artifacts=report['artifacts'],
                    independent_data_status='DECLARED_NOT_AVAILABLE')
    raise ValueError('Unsupported M12 scientific control task')
