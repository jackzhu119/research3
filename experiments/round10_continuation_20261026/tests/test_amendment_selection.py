"""Scientific behavior of the retrospective R10 paired-gain amendment."""
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from paper3r10.amendment_selection import amended_development_decision


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def source_objects():
    people = sorted(['S' + str(i) for i in range(8)],
                    key=lambda p: digest(['r10-mechanism-v1', 'participant', p]))
    tasks = []
    for amplitude in (.3, 1., 3.):
        for fraction in (.01, .1, .5):
            candidate = f'a{amplitude:g}-c{fraction:g}'
            tasks.append(dict(key=candidate, method='profile', candidate_id=candidate,
                              family='evaluate', config=dict(amplitude=amplitude,
                              evidence_fraction=fraction, mode='robust', use_null_completion=True,
                              ridge=1.), session_subset={'salt': 'r10-mechanism-v1'}))
    tasks.append(dict(key='ea', method='support_ea', family='evaluate', config={}))
    plan = dict(stage='mechanism', tasks=tasks, synthetic_fixture=True,
                recipe_sha256='r' * 64, study_registry={'sha256': 'a' * 64})
    rows = []
    for task in tasks:
        for index, person in enumerate(people):
            base = [.25, .4, .65, .75, .55, .6, .5, .65][index]
            gain = 0.
            if task.get('candidate_id') == 'a0.3-c0.01':
                gain = [.02, .03, .01, .02][index] if index < 4 else .01
            if task.get('candidate_id') == 'a0.3-c0.1':
                gain = .018 if index < 4 else .01
            row = dict(method=task['method'], candidate_id=task.get('candidate_id', ''),
                       configuration_id=digest(task['config']), task_key=task['key'],
                       cohort='meta_validation', role='meta_validation', seed=20261018,
                       budget=20, participant_id=person, session_id='one',
                       selection_split='selection' if index < 4 else 'check',
                       n_query=100, ba=base + gain, balanced_nll=.68 - gain,
                       updated_query_fraction=.5 if gain else 0., certificate_violation_count=0,
                       numerical_fallback_count=0, incremental_prediction_seconds=.1,
                       query_signature=digest([person, 'Q']), source_identity_sha256='b' * 64,
                       data_identity_sha256='c' * 64)
            rows.append(row)
    coverage = dict(total_tasks=len(tasks), sealed_complete_tasks=len(tasks),
                    metric_expected_tasks=len(tasks), metric_valid_tasks=len(tasks),
                    invalid_compact_metrics=[])
    return plan, rows, coverage, []


def decide(objects):
    return amended_development_decision(*objects, registry_sha256='a' * 64,
                                        source_plan_sha256='d' * 64, source_report_sha256='e' * 64)


def test_paired_band_can_choose_lower_mean_conservative_candidate_not_hard_argmax():
    objects = source_objects(); before = copy.deepcopy(objects)
    result = decide(objects)
    assert result['status'] == 'EXPLORATORY_SELECTED' and result['allow_formal_exploratory']
    assert result['best_selection_candidate'] == 'a0.3-c0.01'
    assert result['selected_candidate_id'] == 'a0.3-c0.1'
    assert result['eligible_candidate_ids'] == ['a0.3-c0.01', 'a0.3-c0.1']
    best = next(c for c in result['candidates'] if c['candidate_id'] == 'a0.3-c0.01')
    assert best['paired_ba_gain_se'] == pytest.approx(math.sqrt(.0002 / 3) / 2)
    assert result['legacy_decision_reconstruction']['selected_candidate_id'] == 'a0.3-c0.5'
    assert result['legacy_decision_reconstruction']['status'] == 'STOP'
    assert result['retrospective_amendment'] and not result['true_future_risk_guaranteed']
    assert objects == before  # no source object or historical receipt is mutated


def test_participant_specific_common_difficulty_shift_does_not_change_paired_choice():
    objects = source_objects(); before = decide(objects)
    people = before['selection_participant_ids']
    shifts = dict(zip(people, [.10, .05, -.05, -.10]))
    for row in objects[1]:
        row['ba'] += shifts.get(row['participant_id'], 0.)
    after = decide(objects)
    assert after['selected_candidate_id'] == before['selected_candidate_id']
    assert after['one_se_paired_gain_cutoff'] == pytest.approx(before['one_se_paired_gain_cutoff'])
    for a, b in zip(before['candidates'], after['candidates']):
        assert a['paired_ba_gain'] == pytest.approx(b['paired_ba_gain'])
        assert a['paired_ba_gain_se'] == pytest.approx(b['paired_ba_gain_se'])


def test_adverse_check_and_old_cost_thresholds_are_diagnostics_never_rerank_or_stop():
    objects = source_objects(); before = decide(objects)
    for row in objects[1]:
        if row['method'] == 'profile':
            row['incremental_prediction_seconds'] = 10.
            if row['selection_split'] == 'check':
                row['ba'] = .01
                row['balanced_nll'] = 3.
    after = decide(objects)
    assert after['selected_candidate_id'] == before['selected_candidate_id']
    assert after['allow_formal_exploratory'] and after['status'] == 'EXPLORATORY_SELECTED'
    failures = after['diagnostics']['legacy_threshold_failures']
    assert 'check_ba_noninferiority' in failures and 'check_nll_guard' in failures
    assert 'incremental_prediction_cost' in failures
    assert not after['performance_gate_blocks_exploration']


@pytest.mark.parametrize('gain', [0., -.01])
def test_zero_or_negative_effects_still_continue_honestly_without_forcing_active_candidate(gain):
    objects = source_objects()
    ea = {r['participant_id']: r['ba'] for r in objects[1] if r['method'] == 'support_ea'}
    for row in objects[1]:
        if row['method'] == 'profile':
            row['ba'] = ea[row['participant_id']] + gain
            row['updated_query_fraction'] = 0.
    result = decide(objects)
    assert result['selected_candidate_id'] == 'a0.3-c0.5'
    assert result['allow_formal_exploratory']
    assert result['selection_evidence_label'] == 'NO_POSITIVE_DEVELOPMENT_GAIN'
    assert not result['positive_gain_or_nonzero_update_guaranteed']


def test_string_csv_rows_order_and_query_counts_do_not_change_participant_equal_choice():
    objects = source_objects(); before = decide(objects)
    for index, row in enumerate(objects[1]):
        row['n_query'] = index + 1
        for key in ('seed', 'budget', 'n_query', 'ba', 'balanced_nll', 'updated_query_fraction',
                    'certificate_violation_count', 'numerical_fallback_count', 'incremental_prediction_seconds'):
            row[key] = str(row[key])
    objects[1].reverse(); objects[0]['tasks'].reverse()
    result = decide(objects)
    assert result['selected_candidate_id'] == before['selected_candidate_id']
    assert result['one_se_paired_gain_cutoff'] == pytest.approx(before['one_se_paired_gain_cutoff'])


@pytest.mark.parametrize('damage', ['missing_pair', 'wrong_identity', 'duplicate', 'nan', 'changed_split', 'partial'])
def test_invalid_paired_data_are_reported_without_intersection_dropping(damage):
    objects = source_objects(); row = objects[1][0]
    if damage == 'missing_pair': objects[1].pop(0)
    if damage == 'wrong_identity': row['query_signature'] = 'z' * 64
    if damage == 'duplicate': objects[1].append(dict(row))
    if damage == 'nan': row['ba'] = float('nan')
    if damage == 'changed_split': row['selection_split'] = 'check'
    if damage == 'partial': objects[2]['metric_valid_tasks'] -= 1
    result = decide(objects)
    assert result['status'] == 'BLOCKED_INVALID_DEVELOPMENT_DATA'
    assert not result['allow_formal_exploratory'] and result['stop_reasons']


def test_module_import_needs_only_stdlib_and_does_not_create_files(tmp_path):
    package = str(Path(__file__).resolve().parents[1])
    code = ('import sys;sys.path.insert(0, ' + repr(package) + ');'
            'import paper3r10.amendment_selection;'
            'assert not any(k.startswith(("numpy", "torch")) for k in sys.modules)')
    result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())
