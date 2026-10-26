"""Independent R10 continuation checks; generated CPU-only software evidence.

Legacy metadata marked REAL_DATA below deliberately exercises the real reader's
wire contracts; it is generated test metadata, never actual EEG efficacy data.
The final test executes the native EEGNet/cache/head path on generated signals.
No GPU, Slurm, network writes, or old scientific execution is performed.
"""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest
import torch

from paper3r10 import continuation, study
from paper3r10.amendment_selection import amended_development_decision
from paper3r10.platform import common, runtime
from paper3r10.report import generate_report, write_csv


def package_fixture(path):
    (path / 'paper3r10/platform').mkdir(parents=True)
    (path / 'START.py').write_text('# software metadata fixture only\n')
    (path / 'paper3r10/algorithm.py').write_text('# pinned old software fixture\n')
    (path / 'paper3r10/platform/entry.py').write_text('# never executed by these metadata tests\n')
    return path


def snapshot(path):
    return {p.relative_to(path).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in path.rglob('*') if p.is_file()}


def legacy_fixture(tmp_path):
    root = tmp_path / 'school'; root.mkdir()
    old_package = package_fixture(tmp_path / 'old-package')
    new_package = package_fixture(tmp_path / 'new-package')
    payload = root / 'metadata-only.json'; common.atomic(payload, {'generated_software_fixture': True})
    item = study.describe(payload, root)
    registry = {'schema_version': 'paper3-r10-study-registry-v1', 'root': str(root),
        'source_by_seed': {str(seed): {'source_receipt': item, 'reference_receipt': item}
                           for seed in study.SEEDS},
        'external_manifests': {'pan2023': item, 'weibo2014': item},
        'baseline_artifacts': {}, 'previously_used_participants': ['OLD:software-fixture']}
    registry_path = root / 'registry.json'; common.atomic(registry_path, registry)
    descriptor = study.describe(registry_path, root)
    plan = study.build_plan(root, descriptor, package=old_package, stage='mechanism')
    output = Path(plan['output']); output.mkdir(parents=True)
    plan_path = study.write_plan(plan, output / 'plan.json')
    people = sorted(['P' + str(i) for i in range(8)],
        key=lambda p: common.digest(['r10-mechanism-v1', 'participant', p]))
    for task in plan['tasks']:
        folder = common.task_folder(plan, task); folder.mkdir()
        source = {'checkpoint': 'generated-source-' + str(task['seed']),
                  'source_receipt_sha256': item['sha256'], 'reference_receipt_sha256': item['sha256']}
        data = {'split': 'generated-meta-validation-8'}
        payload = folder / 'predictions.bin'; payload.write_bytes(b'not-a-real-prediction')
        paths = [payload]; fields = {}
        if task['family'] == 'feature_reference':
            arrays = folder / 'reference.npz'
            np.savez(arrays, ea_mean=np.zeros(2), ea_whitener=np.eye(2), ea_lengthscale=np.asarray(1.),
                     raw_mean=np.zeros(2), raw_whitener=np.eye(2), raw_lengthscale=np.asarray(1.))
            metadata = folder / 'reference.json'
            common.atomic(metadata, {'schema_version': 'paper3-r10-feature-reference-v1',
                'seed': task['seed'], 'role': 'meta_train', 'source_identity': source,
                'catalog_identity': data, 'arrays': study.describe(arrays, folder),
                'references': {view: {'provenance': {'role': 'meta_train',
                    'registry_sha256': descriptor['sha256']}} for view in ('ea', 'raw')}})
            paths.extend([arrays, metadata])
        else:
            rows = []
            for i, person in enumerate(people):
                baseline = [.25, .4, .65, .75, .55, .6, .5, .65][i]
                candidate = task.get('candidate_id', '')
                gain = ([.02, .03, .01, .02][i] if i < 4 else .01) if candidate == 'a0.3-c0.01' else (
                    .018 if candidate == 'a0.3-c0.1' else 0.)
                for budget in task['budgets']:
                    rows.append({'participant_id': person, 'session_id': 'one', 'seed': task['seed'],
                        'budget': budget, 'method': task['method'], 'role': 'meta_validation',
                        'cohort': 'meta_validation', 'selection_split': 'selection' if i < 4 else 'check',
                        'n_query': 100, 'n_class0': 50, 'n_class1': 50,
                        'ba': baseline + gain, 'balanced_nll': .68 - gain, 'balanced_brier': .24,
                        'n_valid_support': budget, 'elapsed_seconds': .2,
                        'fit_seconds': .01, 'updated_query_fraction': .5 if gain else 0.,
                        'certificate_violation_count': 0, 'numerical_fallback_count': 0,
                        'incremental_prediction_seconds': .01,
                        'query_signature': common.digest([person, 'same-B20-Q']),
                        'source_identity_sha256': common.digest(source),
                        'data_identity_sha256': common.digest(data)})
            metrics = folder / 'session_metrics.csv'; write_csv(metrics, rows); paths.append(metrics)
            fields.update(sessions=8, participants=8, completed_session_budget_cells=len(rows))
        result = {'status': 'COMPLETE', 'scientific_scope': 'REAL_DATA',
            'recipe_sha256': plan['recipe_sha256'], 'task_key': task['key'],
            'task_identity': common.task_identity(plan, task), 'source_identity': source,
            'catalog_identity': data, 'registry_sha256': descriptor['sha256'],
            'artifacts': [study.describe(p, folder) for p in paths], **fields}
        common.atomic(folder / 'result.json', result); common.seal_result(plan, task)
    report = generate_report(plan_path)
    assert report['status'] == 'COMPLETE' and len(plan['tasks']) == 15
    return root, old_package, new_package, plan_path, plan


def inspect_fixture(path):
    return continuation.inspect_legacy(path, expected_recipe=None, expected_plan_sha256=None)


def test_legacy_compact_reader_ignores_absent_old_code_and_never_reads_predictions(tmp_path, monkeypatch):
    root, old_package, _, path, plan = legacy_fixture(tmp_path)
    before = snapshot(Path(plan['output']))
    (old_package / 'paper3r10/algorithm.py').unlink()
    original = Path.read_bytes
    def no_old_payload(self):
        if self.name == 'predictions.bin':
            raise AssertionError('Continuation must not rehash full historical prediction payloads')
        return original(self)
    monkeypatch.setattr(Path, 'read_bytes', no_old_payload)
    captured = inspect_fixture(path)
    assert captured['proof']['legacy_mechanism_completed_tasks'] == 15
    assert captured['proof']['source_code_manifest_sha256'] == plan['code_manifest_sha256']
    assert captured['proof']['legacy_physical_code_imported'] is False
    assert captured['proof']['full_predictions_rehashed'] is False
    monkeypatch.setattr(Path, 'read_bytes', original)
    assert snapshot(Path(plan['output'])) == before


def test_paired_amendment_check_four_cannot_rerank_or_create_original_pass(tmp_path):
    _, _, _, path, _ = legacy_fixture(tmp_path)
    captured = inspect_fixture(path)
    def decide(rows):
        return amended_development_decision(captured['plan'], rows, captured['coverage'], captured['ledger'],
            registry_sha256=captured['registry']['sha256'],
            source_plan_sha256=captured['proof']['source_plan_sha256'],
            source_report_sha256=captured['proof']['source_report_sha256'])
    clean = decide(captured['rows']); bad = deepcopy(captured['rows'])
    for row in bad:
        if row['selection_split'] == 'check' and row['method'] == 'profile':
            row.update(ba='.001', balanced_nll='9', incremental_prediction_seconds='100')
    adverse = decide(bad)
    assert clean['status'] == adverse['status'] == 'EXPLORATORY_SELECTED'
    assert clean['selected_candidate_id'] == adverse['selected_candidate_id'] == 'a0.3-c0.1'
    assert clean['one_se_paired_gain_cutoff'] == adverse['one_se_paired_gain_cutoff']
    assert adverse['allow_formal_exploratory'] and not adverse['performance_gate_blocks_exploration']
    assert adverse['legacy_decision_reconstruction']['status'] == 'STOP'
    assert adverse['legacy_decision_reconstruction']['reconstruction_only_not_a_historical_school_execution_receipt']
    assert adverse['historical_check_outcomes_already_observed'] and adverse['retrospective_amendment']
    assert not adverse['true_future_risk_guaranteed']


def test_actual_continuation_prepares_165_without_submit_and_reuses_legacy_scan_once(tmp_path, monkeypatch):
    root, _, package, old_path, old_plan = legacy_fixture(tmp_path)
    old_output = Path(old_plan['output']); before = snapshot(old_output)
    path, prepared = continuation.prepare_continuation(root, source_plan=old_path, package=package,
        expected_recipe=None, expected_plan_sha256=None)
    plan = common.load_plan(path)[1]
    assert len(plan['tasks']) == 165 and prepared['submitted_jobs'] == 0
    assert plan['resources']['memory_per_job'] == '64G'
    assert plan['resources']['jobs'] == 4 and plan['resources']['gpus_per_job'] == 1
    assert plan['resources']['cpus_per_job'] == 16
    assert plan['stage'] == 'formal_exploratory' and plan['retrospective_amendment']
    assert not plan['performance_gate_controls_training'] and 'freeze' not in plan
    assert plan['legacy_mechanism_rerun_tasks'] == 0
    imported = [t for t in plan['tasks'] if t.get('legacy_reference_reuse')]
    assert len(imported) == 1 and imported[0]['seed'] == 20261018
    assert imported[0]['legacy_reference_reuse']['source_recipe_sha256'] == old_plan['recipe_sha256']
    assert imported[0]['legacy_reference_reuse']['reference_npz']['sha256'] == common.sha(
        old_output / imported[0]['key'] / 'reference.npz')
    old_evaluations = {t['key'] for t in old_plan['tasks'] if t['family'] != 'feature_reference'}
    assert not old_evaluations.intersection(t['key'] for t in plan['tasks'])
    assert not (Path(plan['output']) / 'runtime/SUBMIT_AUTHORIZATION.json').exists()
    assert all(t.get('optional_baseline') for t in plan['tasks'] if t['method'] in study.NEEDS_ARTIFACTS)
    state = common.read(Path(plan['output']) / 'runtime/tasks.json')['tasks']
    assert sum(row['phase'] == 'BLOCKED_CONFIGURATION' for row in state.values()) == 63
    report = generate_report(path)
    assert report['status'] == 'PARTIAL' and report['retrospective_protocol_revision']
    assert not report['independent_confirmation']
    assert report['primary_family_size'] == 12 and report['secondary_ablation_family_size'] == 24
    assert all(not r.get('significant_holm_05') for r in report['comparisons'])
    claimed = runtime.claim_task(plan, 0, 'CPU_METADATA_ONLY')
    assert claimed and claimed[0]['method'] == 'feature_reference'; claimed[1].close()
    assert snapshot(old_output) == before
    monkeypatch.setattr(continuation, 'inspect_legacy', lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError('Existing amended plan must not rescan all original15 results')))
    again, result = continuation.prepare_continuation(root, source_plan=old_path, package=package,
        expected_recipe=None, expected_plan_sha256=None)
    assert again == path and result['reused_continuation_plan'] and snapshot(old_output) == before


def test_existing_unamended_plan_cannot_be_mislabeled_as_continuation_resume(tmp_path):
    root, _, package, source_path, source = legacy_fixture(tmp_path)
    foreign = deepcopy(source)
    output = root / 'runs/r10-foreign-mechanism'
    foreign.update(output=str(output), package=str(package),
                   code_manifest=study.production_manifest(package))
    foreign['resources']['memory_per_job'] = '64G'
    foreign = common.sign_plan(foreign)
    study.write_plan(foreign, output / 'plan.json')
    before = snapshot(Path(source['output']))
    with pytest.raises(ValueError, match='continuation|amend|exploratory'):
        continuation.prepare_continuation(root, source_plan=source_path, output=output,
            package=package, expected_recipe=None, expected_plan_sha256=None)
    assert snapshot(Path(source['output'])) == before


def test_legacy_reference_import_copies_exact_bytes_before_any_source_or_gpu_open(tmp_path, monkeypatch):
    from paper3r10 import worker
    root, _, package, path, old = legacy_fixture(tmp_path)
    target, _ = continuation.prepare_continuation(root, source_plan=path, package=package,
        expected_recipe=None, expected_plan_sha256=None)
    plan = common.load_plan(target)[1]
    task = next(t for t in plan['tasks'] if t.get('legacy_reference_reuse'))
    def forbidden(*_a, **_k):
        raise AssertionError('Exact reference import must not open a source model, catalog or GPU')
    monkeypatch.setattr(worker, 'load_source', forbidden)
    monkeypatch.setattr(worker, 'open_catalog', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    before = snapshot(Path(old['output']))
    result = worker.run_task(plan, task, 'cpu')
    folder = common.task_folder(plan, task)
    for name in ('reference.json', 'reference.npz'):
        assert (folder / name).read_bytes() == (Path(old['output']) / task['key'] / name).read_bytes()
    assert result['legacy_reference_reuse']['reuse_kind'] == 'EXACT_BYTE_COPY_NO_RETRAINING'
    assert result['legacy_reference_reuse']['source_catalog_or_gpu_opened'] is False
    assert snapshot(Path(old['output'])) == before


def test_real_cpu_additional_seeds_head_baseline_and_fixed_future_query():
    import test_native_workflow as native_fixtures
    from paper3r10 import worker
    from paper3r10.data import load_source, open_catalog
    case = native_fixtures.NativeWorkflowTests('runTest')
    case.setUpClass(); case.setUp()
    try:
        for seed in (20261019, 20261020):
            source, geometry, identity = load_source(case.plan, case.registry, seed, 'cpu')
            catalog = open_catalog(case.plan, case.registry, 'holdout')
            task = {'key': 'independent-head-' + str(seed), 'family': 'evaluate', 'method': 'head_rank4',
                'method_config': {}, 'config': {}, 'seed': seed, 'cohort': 'holdout',
                'role': 'internal_holdout', 'budgets': [4, 20]}
            case.plan['tasks'].append(task)
            folder = Path(case.plan['output']) / task['key']; folder.mkdir(parents=True)
            result = worker.evaluate_task(case.plan, task, folder, source, geometry, catalog,
                identity, 'd' * 64, 'cpu', case.registry)
            assert result['completed_session_budget_cells'] == 8
            predictions = []
            for path in folder.glob('sessions/*.predictions.npz'):
                with np.load(path, allow_pickle=False) as saved:
                    assert np.isfinite(saved['probabilities']).all()
                    predictions.append(tuple(saved['event_uids'].tolist()))
                    assert len(saved['event_uids']) == 25
            assert len(predictions) == 8
            assert all(predictions.count(uids) == 2 for uids in set(predictions))
            assert not source.training and all(not p.requires_grad for p in source.parameters())
    finally:
        case.tearDown(); case.tearDownClass()
