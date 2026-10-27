"""Offline specification/evidence audit. Never starts EEG, Slurm or networking."""
from pathlib import Path
import csv
import hashlib
import json
import math
from statistics import NormalDist

ROOT = Path(__file__).resolve().parent


def main():
    config = json.loads((ROOT/'EXPERIMENT_CONFIG.json').read_text())
    assert config['execution_enabled'] is False
    assert config['status'] == 'DESIGN_DRAFT_REQUIRES_DEVELOPMENT_FREEZE'
    assert config['data']['acquired_budgets'] == [4, 8, 12, 16, 20]
    assert config['data']['primary_budget'] == 20
    assert config['data']['reference_participants'] + config['data']['pilot_participants'] == 40
    assert config['data']['reference_participants'] * config['data']['reference_events_per_participant'] == 64
    assert config['model']['spatial_dimension'] == 4
    assert config['model']['primary_head_trainable_parameters'] == 0
    assert config['matching']['scale_range'] == [0, 1]
    assert config['matching']['EA_zero_anchor_in_minimum'] is False
    assert config['data']['target_query_inputs_used_for_fit_or_scaling'] is False
    assert config['data']['target_query_labels_used_for_selection'] is False
    with (ROOT/'PRIMARY_COMPARISONS.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert rows == config['statistics']['primary_family']
    assert len(rows) == 6 and len({row['id'] for row in rows}) == 6
    assert sum(row['metric'] == 'BA' for row in rows) == 4
    assert sum(row['metric'] == 'balanced_NLL' for row in rows) == 2
    assert config['resources']['workers'] == 4
    assert config['resources']['gpus_per_worker'] == 1
    assert config['resources']['cpus_per_worker'] == 16
    assert config['resources']['memory_per_worker'] == '64G'
    assert config['resources']['automatic_submission'] is False
    assert not any(config['operations'][key] for key in [
        'automatic_child_refresh', 'scientific_task_signals_from_monitor',
        'monitor_failure_controls_training', 'network_failure_controls_training',
        'full_source_revalidation_every_fit'])
    assumptions = config['statistics']['power_planning_assumptions']
    quantile = NormalDist().inv_cdf(1 - .05/(2*len(rows)))
    power_quantile = NormalDist().inv_cdf(assumptions['power'])
    count = math.ceil(((quantile+power_quantile)*assumptions['participant_difference_sd']
                       /assumptions['BA_difference'])**2)
    assert count == assumptions['normal_approximation_complete_n'] == 87
    manifest = json.loads((ROOT/'evidence/MANIFEST.json').read_text())
    checked = 0
    for name, row in manifest['files'].items():
        data = (ROOT/'evidence'/name).read_bytes()
        assert len(data) == row['bytes']
        assert hashlib.sha256(data).hexdigest() == row['sha256']
        checked += 1
    with (ROOT/'evidence/R10_comparisons.csv').open(newline='') as handle:
        comparisons = list(csv.DictReader(handle))
    selected = [row for row in comparisons if row.get('name') == 'profile_vs_ea'
                and row.get('budget') == '20' and row.get('metric') == 'ba']
    assert len(selected) == 3
    assert all(float(row['holm_pvalue']) == 1 for row in selected)
    assert all(row['selection_was_retrospectively_revised'] == 'True' for row in selected)
    for name in ['MECHANISM_EXPERIMENT_ZH.md', 'STATISTICS_ZH.md']:
        assert (ROOT/name).is_file()
    result = dict(status='DESIGN_AND_EVIDENCE_AUDIT_PASSED', verified_evidence_files=checked,
                  primary_comparisons=len(rows), illustrative_BA_power_complete_n=count,
                  actual_gpu_used=False, slurm_calls=0, new_EEG_experiments_run=0,
                  experiment_engine_implemented=False,
                  scope='Specification and copied evidence consistency only; not experiment execution or GPU compatibility')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
