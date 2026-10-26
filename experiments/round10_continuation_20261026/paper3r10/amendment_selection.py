"""Pure-stdlib, retrospective exploratory amendment to R10 selection.

This module accepts already verified compact source objects. It neither reads a
plan nor writes a decision, freeze, or historical result. Transport verification
belongs to the continuation loader; participant pairing is checked here too.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import statistics


POLICY = 'paired-ea-gain-one-se-exploratory-amendment-v1'
LEGACY_CRITERIA = {
    'selection_participants': 4, 'check_participants': 4,
    'selection_ba_gain_minimum': .0025, 'check_ba_gain_minimum': 0.,
    'check_nll_excess_maximum': .01, 'minimum_changed_query_fraction': .01,
    'maximum_certificate_violations': 0,
    'maximum_global_numerical_fallback_fraction': .01,
    'maximum_incremental_prediction_seconds_per1000': 5.,
    'query_probability_change_threshold': 1e-4,
}
IDENTITIES = ('query_signature', 'source_identity_sha256', 'data_identity_sha256')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _number(row, key, *, minimum=0., maximum=None, integer=False):
    value = row.get(key)
    if isinstance(value, bool):
        raise ValueError('A numeric compact field is required: ' + key)
    value = float(value)
    if (not math.isfinite(value) or value < minimum
            or (maximum is not None and value > maximum)
            or (integer and int(value) != value)):
        raise ValueError('Invalid compact field: ' + key)
    return int(value) if integer else value


def _candidates(plan):
    tasks = [t for t in plan['tasks'] if t.get('method') == 'profile']
    result = {}
    for task in tasks:
        key, config = task.get('candidate_id'), deepcopy(task.get('config', {}))
        if (not isinstance(key, str) or not key or key in result
                or config.get('amplitude') not in (.3, 1., 3.)
                or config.get('evidence_fraction') not in (.01, .1, .5)
                or config.get('mode') != 'robust'
                or config.get('use_null_completion') is not True):
            raise ValueError('The original fixed nine-candidate development grid is required')
        result[key] = config
    if (len(result) != 9 or {(c['amplitude'], c['evidence_fraction']) for c in result.values()}
            != {(a, c) for a in (.3, 1., 3.) for c in (.01, .1, .5)}):
        raise ValueError('Exactly the original nine development candidates are required')
    shared = {_digest({k: v for k, v in c.items()
                       if k not in ('amplitude', 'evidence_fraction')}) for c in result.values()}
    if len(shared) != 1:
        raise ValueError('An undeclared hyperparameter differs across development candidates')
    return result


def _outcomes(rows, candidates):
    result = []
    for source in rows:
        if source.get('method') not in ('profile', 'support_ea'):
            continue
        row = dict(source)
        if row.get('cohort') != 'meta_validation' or row.get('role') != 'meta_validation':
            raise ValueError('Amended selection may read only original meta_validation outcomes')
        for key in ('participant_id', 'session_id'):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError('A fixed participant/session identity is required')
        for key in ('seed', 'budget', 'n_query'):
            row[key] = _number(row, key, integer=True, minimum=1)
        if row['seed'] != 20261018 or row['budget'] not in (4, 8, 20):
            raise ValueError('Original mechanism seed and budgets are required')
        row['ba'] = _number(row, 'ba', maximum=1.)
        row['balanced_nll'] = _number(row, 'balanced_nll')
        row['updated_query_fraction'] = _number(row, 'updated_query_fraction', maximum=1.)
        for key in ('certificate_violation_count', 'numerical_fallback_count'):
            row[key] = _number(row, key, integer=True)
        row['incremental_prediction_seconds'] = _number(row, 'incremental_prediction_seconds')
        if any(not isinstance(row.get(k), str) or len(row[k]) != 64 for k in IDENTITIES):
            raise ValueError('Compact query/source/data identities are required')
        key = row.get('candidate_id', '')
        expected = candidates.get(key) if row['method'] == 'profile' else {}
        if expected is None or row.get('configuration_id') != _digest(expected):
            raise ValueError('Compact candidate configuration differs from the source plan')
        result.append(row)
    return result


def _people(rows, plan):
    assignment = {}
    for row in rows:
        person, split = row['participant_id'], row.get('selection_split')
        if split not in ('selection', 'check') or assignment.get(person, split) != split:
            raise ValueError('Development selection/check assignments are inconsistent')
        assignment[person] = split
    salts = {str(t.get('session_subset', plan.get('session_subset', {})).get('salt', 'r10-mechanism-v1'))
             for t in plan['tasks'] if t.get('method') in ('profile', 'support_ea')}
    if len(salts) != 1 or len(assignment) != 8:
        raise ValueError('Exactly eight original hash-assigned participants are required')
    ordered = sorted(assignment, key=lambda p: _digest([next(iter(salts)), 'participant', p]))
    selection, check = sorted(ordered[:4]), sorted(ordered[4:])
    if (any(assignment[p] != 'selection' for p in selection)
            or any(assignment[p] != 'check' for p in check)):
        raise ValueError('Development split differs from its predetermined metadata hash')
    return selection, check


def _paired(rows, candidate, metric, participants):
    selected = [r for r in rows if r['budget'] == 20 and r['participant_id'] in participants]
    def collect(method):
        cells = {}
        for row in selected:
            if row['method'] != method or (method == 'profile' and row['candidate_id'] != candidate):
                continue
            key = row['participant_id'], row['session_id'], row['seed']
            if key in cells:
                raise ValueError('Duplicated paired participant/session/seed cell')
            cells[key] = row
        return cells
    left, right = collect('profile'), collect('support_ea')
    if not left or set(left) != set(right):
        raise ValueError('Full paired development cells are required; intersection dropping is forbidden')
    if {k[0] for k in left} != set(participants):
        raise ValueError('A candidate omitted a predetermined development participant')
    for key in left:
        if any(left[key][field] != right[key][field] for field in IDENTITIES):
            raise ValueError('Paired query/source/data identity mismatch')
    a, b = [], []
    for person in participants:
        aa, bb = [], []
        for seed in sorted({k[2] for k in left if k[0] == person}):
            keys = [k for k in left if k[0] == person and k[2] == seed]
            aa.append(statistics.mean(left[k][metric] for k in keys))
            bb.append(statistics.mean(right[k][metric] for k in keys))
        a.append(statistics.mean(aa)); b.append(statistics.mean(bb))
    return a, b


def _mechanisms(rows):
    cells = [r for r in rows if r['method'] == 'profile']
    queries = sum(r['n_query'] for r in cells)
    violations = sum(r['certificate_violation_count'] for r in cells)
    failures = sum(r['numerical_fallback_count'] for r in cells)
    seconds = sum(r['incremental_prediction_seconds'] for r in cells)
    if not queries or failures > queries or violations > queries:
        raise ValueError('Invalid development mechanism event counters')
    return dict(queries=queries, certificate_violations=violations,
                numerical_fallback_count=failures, numerical_fallback_fraction=failures / queries,
                incremental_prediction_seconds=seconds,
                incremental_prediction_seconds_per1000=1000 * seconds / queries)


def _diagnostics(rows, summary, selection, check, mechanisms):
    candidate = summary['candidate_id']
    a, b = _paired(rows, candidate, 'ba', check)
    nll, nll0 = _paired(rows, candidate, 'balanced_nll', check)
    snll, snll0 = _paired(rows, candidate, 'balanced_nll', selection)
    cells = [r for r in rows if r['method'] == 'profile'
             and r['candidate_id'] == candidate and r['budget'] == 20]
    changed = sum(r['n_query'] * r['updated_query_fraction'] for r in cells) / sum(r['n_query'] for r in cells)
    check_gain = statistics.mean(x - y for x, y in zip(a, b))
    check_nll = statistics.mean(x - y for x, y in zip(nll, nll0))
    gates = {
        'selection_ba_gain': summary['paired_ba_gain'] >= .0025 - 1e-12,
        'check_ba_noninferiority': check_gain >= -1e-12,
        'check_nll_guard': check_nll <= .01 + 1e-12,
        'nontrivial_updates': changed >= .01 - 1e-12,
        'certificate_violations': mechanisms['certificate_violations'] == 0,
        'global_numerical_fallback': mechanisms['numerical_fallback_fraction'] <= .01 + 1e-12,
        'incremental_prediction_cost': mechanisms['incremental_prediction_seconds_per1000'] <= 5. + 1e-12,
        'pairing_and_identity': True,
    }
    return dict(legacy_thresholds=gates,
                legacy_threshold_failures=[k for k, passed in gates.items() if not passed],
                selected_selection_ba_gain=summary['paired_ba_gain'],
                selected_check_ba_gain=check_gain, selected_check_nll_excess=check_nll,
                selected_selection_nll_excess=statistics.mean(x - y for x, y in zip(snll, snll0)),
                updated_query_fraction_query_weighted=changed,
                global_mechanisms=mechanisms)


def amended_development_decision(plan, rows, coverage, ledger, *, registry_sha256,
                                 source_plan_sha256=None, source_report_sha256=None):
    """Return a decision from verified source objects without any I/O or imports of NumPy.

    ``rows`` are session_metrics.csv dictionaries, raw strings or typed values;
    ``coverage`` and ``ledger`` are the source report's coverage/source_receipts.
    Outcome failures are diagnostic. Missing/corrupt paired data remain errors.
    """
    result = dict(schema_version='paper3-r10-amended-development-decision-v1',
                  status='BLOCKED_INVALID_DEVELOPMENT_DATA', allow_formal_exploratory=False,
                  selection_policy=POLICY, retrospective_amendment=True,
                  original_development_results_already_observed=True,
                  scientific_scope='SOFTWARE_TEST_ONLY' if plan.get('synthetic_fixture') else 'REAL_DATA',
                  inference_scope='exploratory_post_result_protocol_amendment',
                  source_plan_recipe_sha256=plan.get('recipe_sha256'),
                  source_plan_sha256=source_plan_sha256, source_report_sha256=source_report_sha256,
                  study_registry_sha256=registry_sha256, source_result_receipts=deepcopy(ledger),
                  coverage=deepcopy(coverage),
                  check_role='diagnostic_only_never_rerank_or_block_exploration',
                  performance_gate_blocks_exploration=False,
                  target_query_labels_used_for_selection=False, development_query_labels_used=True,
                  historical_check_outcomes_already_observed=True,
                  historical_target_results_used_for_algorithm_design=True,
                  selection_uses_check_outcomes=False,
                  new_independent_confirmation_claimed=False,
                  selection_roles=['meta_validation_selection_four'],
                  true_future_risk_guaranteed=False, positive_gain_or_nonzero_update_guaranteed=False,
                  original_results_must_remain_immutable=True)
    try:
        if plan.get('stage') != 'mechanism':
            raise ValueError('Target/exploratory/confirmation plans cannot select the amended configuration')
        if (not isinstance(registry_sha256, str) or len(registry_sha256) != 64
                or plan.get('study_registry', {}).get('sha256') != registry_sha256):
            raise ValueError('Source registry identity differs')
        if (coverage['total_tasks'] != len(plan['tasks'])
                or coverage['metric_expected_tasks'] != sum(t.get('family') != 'feature_reference' for t in plan['tasks'])
                or coverage['sealed_complete_tasks'] != coverage['total_tasks']
                or coverage['metric_valid_tasks'] != coverage['metric_expected_tasks']
                or coverage['invalid_compact_metrics']):
            raise ValueError('All original development compact metrics must be sealed and complete')
        candidates = _candidates(plan)
        outcomes = _outcomes(rows, candidates)
        selection, check = _people(outcomes, plan)
        summaries = []
        for key, config in sorted(candidates.items()):
            _paired(outcomes, key, 'ba', selection + check)
            _paired(outcomes, key, 'balanced_nll', selection + check)
            a, b = _paired(outcomes, key, 'ba', selection)
            delta = [x - y for x, y in zip(a, b)]
            summaries.append(dict(candidate_id=key, configuration=config, configuration_id=_digest(config),
                                  mean_ba=statistics.mean(a), se_ba=statistics.stdev(a) / math.sqrt(len(a)),
                                  paired_ba_gain=statistics.mean(delta),
                                  paired_ba_gain_se=statistics.stdev(delta) / math.sqrt(len(delta)),
                                  selection_participant_differences=dict(zip(selection, delta)),
                                  n_participants=len(delta)))
        best = min(summaries, key=lambda c: (-c['paired_ba_gain'], c['candidate_id']))
        cutoff = best['paired_ba_gain'] - best['paired_ba_gain_se']
        eligible = [c for c in summaries if c['paired_ba_gain'] >= cutoff - 1e-12]
        conservative = lambda c: (-c['configuration']['evidence_fraction'],
                                  c['configuration']['amplitude'], c['candidate_id'])
        selected = min(eligible, key=conservative)
        mechanisms = _mechanisms(outcomes)
        diagnostics = _diagnostics(outcomes, selected, selection, check, mechanisms)
        old_best = min(summaries, key=lambda c: (-c['mean_ba'], c['candidate_id']))
        old_cutoff = old_best['mean_ba'] - old_best['se_ba']
        old_eligible = [c for c in summaries if c['mean_ba'] >= old_cutoff - 1e-12]
        old_selected = min(old_eligible, key=conservative)
        old_diagnostics = _diagnostics(outcomes, old_selected, selection, check, mechanisms)
        legacy = dict(status='STOP' if old_diagnostics['legacy_threshold_failures'] else 'PASS',
                      reconstruction_only_not_a_historical_school_execution_receipt=True,
                      rule='original_absolute_BA_one_SE_high_c_low_amplitude',
                      selected_candidate_id=old_selected['candidate_id'],
                      best_selection_candidate=old_best['candidate_id'], one_se_cutoff=old_cutoff,
                      eligible_candidate_ids=[c['candidate_id'] for c in old_eligible], **old_diagnostics)
        result.update(status='EXPLORATORY_SELECTED', allow_formal_exploratory=True,
                      selection_participant_ids=selection, check_participant_ids=check,
                      candidates=summaries, best_selection_candidate=best['candidate_id'],
                      one_se_paired_gain_cutoff=cutoff,
                      eligible_candidate_ids=[c['candidate_id'] for c in eligible],
                      conservative_candidate_order=[c['candidate_id'] for c in sorted(eligible, key=conservative)],
                      selected_candidate_id=selected['candidate_id'],
                      selected_configuration=selected['configuration'], configuration_id=selected['configuration_id'],
                      selected_paired_ba_gain=selected['paired_ba_gain'],
                      selected_paired_ba_gain_se=selected['paired_ba_gain_se'],
                      selection_evidence_label='POSITIVE_DEVELOPMENT_MEAN_NOT_CONFIRMATION'
                      if selected['paired_ba_gain'] > 0 else 'NO_POSITIVE_DEVELOPMENT_GAIN',
                      diagnostics=diagnostics, legacy_decision_reconstruction=legacy,
                      rule='mean selection-four paired BA gain; best gain minus its ddof=1 paired SE; '
                           'inside band evidence_fraction descending, amplitude ascending, candidate_id lexical',
                      stop_reasons=[])
    except (ValueError, KeyError, TypeError, ZeroDivisionError, statistics.StatisticsError) as error:
        result['stop_reasons'] = [str(error)]
    result['decision_identity'] = _digest(result)
    return result
