from __future__ import annotations
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from paper3m12 import plan as p
from paper3m12 import native_data as n
from paper3m12.common import atomic_json, load_plan
from test_native_data import metadata_study


def test_registered_full_dag_has_tau_barrier_and_non_gated_secondary(metadata_study):
    root, output, _ = metadata_study
    config = json.loads((PACKAGE/'EXPERIMENT_CONFIG.json').read_text())
    inputs = n.freeze_inputs(root, output, config)
    plan = p.build_plan(root, output, PACKAGE, config, inputs,
                        external_people={'pan2023':['P'+str(i) for i in range(14)],
                                         'weibo2014':['W'+str(i) for i in range(10)]})
    assert len(plan['tasks']) == 217
    by_key = {task['key']:task for task in plan['tasks']}
    assert len(by_key['freeze-tau']['dependencies']) == 3
    assert len(by_key['pilot-decision']['dependencies']) == 24
    assert len(plan['expected_validation_tasks']) == 33
    assert len(plan['expected_secondary_tasks']) == 48
    assert len(plan['expected_exploratory_tasks']) == 105
    for task in plan['tasks']:
        if task['kind'] == 'calibration':
            assert task['target_query_access'] is False
        if task['kind'] == 'paired_block':
            assert 'freeze-tau' in task['dependencies']
        if task['kind'] in {'head_block','shift_block'}:
            assert 'gate' not in task and 'pilot-decision' in task['dependencies']
    assert plan['config']['execution_enabled'] is True
    assert plan['config']['model']['dictionary_sha256'][str(n.SEEDS[0])]
    assert config['execution_enabled'] is False  # historical design retained
    atomic_json(output/'plan.json', plan)
    assert load_plan(output/'plan.json')[1]['recipe_sha256'] == plan['recipe_sha256']


def test_metadata_prepare_is_idempotent_while_original_rounds_advance(metadata_study):
    root, output, _ = metadata_study
    old = root/'runs/training_campaign_parallel_v1/status.json'
    atomic_json(old, {'progress':1})
    first = p.prepare(root, output, PACKAGE)
    sealed = (output/'plan.json').read_bytes()
    atomic_json(old, {'progress':2})
    second = p.prepare(root, output, PACKAGE)
    assert second['status'] == 'EXISTING_M12_PLAN_REUSED'
    assert second['recipe_sha256'] == first['recipe_sha256']
    assert sealed == (output/'plan.json').read_bytes()
    assert json.loads(old.read_text()) == {'progress':2}
    assert first['jobs_submitted'] == second['jobs_submitted'] == 0
    assert first['GPU_used'] is False


def test_optional_external_absence_keeps_core_executable(metadata_study):
    root, output, _ = metadata_study
    result = p.prepare(root, output, PACKAGE)
    plan = load_plan(result['plan'])[1]
    assert set(plan['unavailable_external']) == {'pan2023','weibo2014'}
    assert len(plan['expected_pilot_tasks']) == 24
    assert len(plan['expected_exploratory_tasks']) == 33
    assert plan['independent_confirmation']['status'] == 'NOT_AVAILABLE'


def test_existing_round_destination_refused_before_any_write(metadata_study):
    root, _, _ = metadata_study
    for name in ('training_campaign_parallel_v1', 'r11_campaign'):
        path = root/'runs'/name
        with pytest.raises(ValueError, match='own directory'):
            p.prepare(root, path, PACKAGE)
        assert not path.exists()


def test_real_isolated_bootstrap_prepares_without_scientific_import(metadata_study):
    root, output, registry = metadata_study
    registry_path = root/'cache/registry.json'
    atomic_json(registry_path, registry)
    atomic_json(root/'runs/r11_campaign/plan.json',
                {'study_registry':n.describe(registry_path, root)})
    command = [sys.executable, '-I', '-S', '-B', '-u', str(PACKAGE/'paper3m12/entry.py'),
               'prepare', '--root', str(root), '--output', str(output)]
    first = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)['status'] == 'M12_PLAN_READY_NO_GPU_ALLOCATED'
    assert not (output/'runtime').exists()
    second = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)['status'] == 'EXISTING_M12_PLAN_REUSED'
