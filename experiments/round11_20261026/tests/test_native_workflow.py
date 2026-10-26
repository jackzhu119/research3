"""Real artifact formats and native EEGNet/cache paths with generated test signals.

These exercise production readers, not scientific performance or real EEG data.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from paper3r11 import study, worker
from paper3r11.data import (SessionAccess, describe, load_source, native, open_catalog,
                           clear_source_cache, training_pairs, training_prefixes)
from paper3r11.platform.common import digest, seal_result, sign_plan
from paper3r11.registry import build_registry


class NativeWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        clear_source_cache()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.seed = 20261018
        contract = {'n_times': 128, 'synthetic_fixture': True}
        self.put('contracts/input_contract.json', contract)
        model_class = native('p3work.f0_bridge').CompactEEGNet
        for seed in (20261018, 20261019, 20261020):
            torch.manual_seed(seed)
            checkpoint = self.root / f'fixtures/source-{seed}.pt'
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model': model_class(11, 128).state_dict(), 'config': {'dropout': .25}}, checkpoint)
            receipt = self.put(f'runs/source_training_v2/f0_receipt_seed-{seed}.json',
                {'status': 'SOURCE_F0_COMPLETE', 'pipeline_seed': seed,
                 'signal_contract_sha256': digest(contract), 'checkpoint': describe(checkpoint, self.root)})
            ref_array = self.root / f'fixtures/reference-{seed}.npy'
            np.save(ref_array, np.eye(10, dtype=np.float64) / 10)
            self.put(f'runs/training_campaign_v1/references/seed-{seed}/receipt.json',
                {'status': 'SOURCE_REFERENCE_COMPLETE', 'source_f0_receipt_sha256': describe(receipt, self.root)['sha256'],
                 'signal_contract_sha256': digest(contract),
                 'arrays': {'reference': describe(ref_array, self.root)}})
        partitions = {'meta_train': ['S1', 'S2'], 'meta_validation': ['S3', 'S4'],
                      'internal_holdout': ['S5', 'S6']}
        split = self.put('data/split.json', {'partitions': partitions, 'split_sha256': 'b' * 64})
        sessions = []
        rng = np.random.default_rng(19)
        for role, people in partitions.items():
            for person in people:
                for session in (1, 2):
                    values = rng.normal(size=(50, 11, 128)).astype(np.float32)
                    values -= values.mean(1, keepdims=True)
                    array = self.root / f'data/{person}-{session}.npy'
                    np.save(array, values)
                    rows = [{'participant_id': person, 'session_id': str(session),
                             'event_uid': f'{person}-{session}-{i}', 'trial_uid': f'{person}-{session}-{i}',
                             'tasknumber': 1, 'original_trial_index': i, 'cache_index': i,
                             'label': i % 2, 'valid': i not in (2, 5),
                             'physical_run_id': 'prefix' if i < 25 else 'future',
                             'physical_run_complete_verified': True} for i in range(50)]
                    events = self.put(f'data/{person}-{session}-events.json', rows)
                    sessions.append({'partition': role, 'participant_id': person, 'session_id': str(session),
                                     'acquisition_order_verified': True, 'X': describe(array, self.root),
                                     'events': describe(events, self.root)})
        manifest = self.put('data/manifest.json', {'status': 'STIEGER_CACHE_COMPLETE',
            'input_contract_sha256': digest(contract), 'participant_split': describe(split, self.root),
            'sessions': sessions})
        self.put('status/tasks.json', {'W3': {'cache_manifest': describe(manifest, self.root)}})
        self.registry = build_registry(self.root)
        self.plan = {'root': str(self.root), 'output': str(self.root / 'r11-test-output'),
                     'stage': 'exploratory', 'synthetic_fixture': True, 'recipe_sha256': 'c' * 64,
                     'reference_strategy': 'build_meta_train_prefix', 'tasks': []}
        worker._STOP_REQUESTED.clear()

    def tearDown(self):
        worker._STOP_REQUESTED.clear()
        clear_source_cache()
        self.temp.cleanup()

    def put(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def test_native_source_all_seeds_and_paired_prefix_metadata(self):
        catalog = open_catalog(self.plan, self.registry, 'meta_train')
        for seed in (20261018, 20261019, 20261020):
            source, geometry, _ = load_source(self.plan, self.registry, seed, 'cpu')
            before = {key: value.clone() for key, value in source.state_dict().items()}
            records = list(training_prefixes(catalog, 'cpu'))
            self.assertEqual(len(records), 4)
            self.assertEqual({record.participant_id for record in records}, {'S1', 'S2'})
            self.assertTrue(all(record.session_weight == .25 for record in records))
            self.assertTrue(all(len(record.support.acquired_event_uids) == 20 for record in records))
            for record in records:
                ea, _ = native('paper3r7.baseline').fit_reference(source, geometry, record.support, 'support_EA')
                features, logits = worker.feature_batch(source, geometry, record.support.x, ea)
                self.assertEqual(features.shape, (18, 64))
                self.assertEqual(logits.shape, (18, 2))
                self.assertTrue(np.isfinite(logits).all())
            self.assertTrue(all(torch.equal(before[key], value) for key, value in source.state_dict().items()))

    def test_native_holdout_ea_and_frozen_all_budgets_fixed_future_query(self):
        source, geometry, identity = load_source(self.plan, self.registry, self.seed, 'cpu')
        catalog = open_catalog(self.plan, self.registry, 'holdout')
        for method in ('support_ea', 'frozen'):
            task = {'key': method, 'family': 'evaluate', 'method': method, 'method_config': {},
                    'seed': self.seed, 'cohort': 'holdout', 'role': 'internal_holdout',
                    'budgets': [4, 8, 12, 16, 20]}
            self.plan['tasks'].append(task)
            folder = Path(self.plan['output']) / method
            folder.mkdir(parents=True)
            result = worker.evaluate_task(self.plan, task, folder, source, geometry,
                catalog, identity, 'd' * 64, 'cpu', self.registry)
            self.assertEqual(result['completed_session_budget_cells'], 20)
            seal_result(self.plan, task)
            for path in folder.glob('sessions/*.predictions.npz'):
                with np.load(path, allow_pickle=False) as prediction:
                    anchor = 'ea_probabilities' if method == 'support_ea' else 'frozen_probabilities'
                    np.testing.assert_array_equal(prediction['probabilities'], prediction[anchor])
                    self.assertEqual(len(prediction['event_uids']), 25)
                    self.assertNotIn('labels', prediction.files)
        access = SessionAccess(catalog, catalog.plans[0], 'cpu')
        self.assertEqual(len(access.support(4).acquired_event_uids), 4)
        self.assertEqual(len(access.support(20).event_uids), 18)
        self.assertEqual(len(access.query_inputs().event_uids), 25)
        with self.assertRaisesRegex(ValueError, 'meta_train participants'):
            access.meta_train_future_labels()

    def test_legal_training_futures_share_exact_prefix_ea_and_never_enter_cache(self):
        source, geometry, identity = load_source(self.plan, self.registry, self.seed, 'cpu')
        catalog = open_catalog(self.plan, self.registry, 'meta_train')
        cache = self.root / 'training-feature-cache'
        pairs = list(training_pairs(source, geometry, catalog, budgets=[4, 20], device='cpu',
                                    cache_root=cache, source_identity=identity))
        self.assertEqual(len(pairs), 8)
        self.assertTrue(all(pair['training_role'] == 'meta_train' for pair in pairs))
        self.assertTrue(all(pair['weight'] == .25 for pair in pairs))
        self.assertEqual(sum(pair['weight'] for pair in pairs if pair['budget'] == 20), 1.)
        for pair in pairs:
            session = next(item for item in catalog.plans if item.participant_id == pair['participant_id']
                           and item.session_id == pair['session_id'])
            access = SessionAccess(catalog, session, 'cpu')
            support = access.support(pair['budget'])
            ea, _ = native('paper3r7.baseline').fit_reference(source, geometry, support, 'support_EA')
            sx, sz = worker.feature_batch(source, geometry, support.x, ea)
            qx, qz = worker.feature_batch(source, geometry, access.query_inputs().x, ea)
            np.testing.assert_array_equal(pair['support_features'], sx)
            np.testing.assert_array_equal(pair['support_logits'], sz)
            np.testing.assert_array_equal(pair['future_features'], qx)
            np.testing.assert_array_equal(pair['future_logits'], qz)
            np.testing.assert_array_equal(pair['future_y'], access.meta_train_future_labels())
            self.assertFalse(set(pair['acquired_support_ids']) & set(pair['future_ids']))
        for path in cache.glob('*/*.npz'):
            with np.load(path, allow_pickle=False) as arrays:
                self.assertFalse({'labels', 'support_y', 'future_y'} & set(arrays.files))
        holdout = open_catalog(self.plan, self.registry, 'holdout')
        with self.assertRaisesRegex(ValueError, 'meta_train role'):
            list(training_pairs(source, geometry, holdout, budgets=[20], device='cpu',
                                cache_root=cache, source_identity=identity))

    def test_operator_training_replay_and_target_evaluation_with_native_inputs(self):
        source, geometry, identity = load_source(self.plan, self.registry, self.seed, 'cpu')
        train = open_catalog(self.plan, self.registry, 'meta_train')
        reference_task = {'key': 'reference', 'family': 'feature_reference', 'role': 'meta_train',
                          'seed': self.seed}
        self.plan['tasks'].append(reference_task)
        reference_folder = Path(self.plan['output']) / 'reference'
        reference_folder.mkdir(parents=True)
        worker.feature_reference_task(self.plan, reference_task, reference_folder, source, geometry,
                                      train, identity, 'd' * 64)
        seal_result(self.plan, reference_task)
        config = {'feature_rank': 4, 'noise_strength': 1., 'ridge': .1,
                  'max_iter': 200, 'tolerance': 1e-8}
        dependencies = {}
        for budget in (4, 20):
            task = {'key': f'train-{budget}', 'family': 'operator_train', 'method': 'operator',
                    'role': 'meta_train', 'seed': self.seed, 'budgets': [budget],
                    'method_config': config, 'reference_dependency': 'reference', 'requires': ['reference']}
            self.plan['tasks'].append(task)
            folder = Path(self.plan['output']) / task['key']
            folder.mkdir()
            result = worker.operator_train_task(self.plan, task, folder, source, geometry,
                                                 train, identity, 'd' * 64)
            self.assertEqual(result['operator_budget'], budget)
            self.assertIn('operator.json', result['paper_artifacts'])
            metadata = json.loads((folder / 'operator.json').read_text())
            self.assertTrue(metadata['future_training_labels_used'])
            self.assertFalse(metadata['target_query_labels_used'])
            self.assertEqual(metadata['training_pair_count'], 4)
            seal_result(self.plan, task)
            dependencies[str(budget)] = task['key']
        task = {'key': 'operator-evaluation', 'family': 'evaluate', 'method': 'operator',
                'role': 'internal_holdout', 'cohort': 'holdout', 'seed': self.seed,
                'budgets': [4, 20], 'method_config': config, 'operator_dependencies': dependencies}
        self.plan['tasks'].append(task)
        folder = Path(self.plan['output']) / task['key']
        folder.mkdir()
        catalog = open_catalog(self.plan, self.registry, 'holdout')
        result = worker.evaluate_task(self.plan, task, folder, source, geometry, catalog,
                                     identity, 'd' * 64, 'cpu', self.registry)
        self.assertEqual(result['completed_session_budget_cells'], 8)
        self.assertEqual(result['resolved_configuration']['feature_rank'], 4)
        with patch('paper3r11.worker.algorithm.adapt', side_effect=AssertionError('Replay must not refit')):
            worker.evaluate_task(self.plan, task, folder, source, geometry, catalog,
                                 identity, 'd' * 64, 'cpu', self.registry)
        for path in folder.glob('sessions/*.predictions.npz'):
            with np.load(path, allow_pickle=False) as prediction:
                self.assertEqual(prediction['probabilities'].shape, (25, 2))
                self.assertNotIn('labels', prediction.files)
        selection_task = {'key': 'selected', 'family': 'selection', 'role': 'meta_validation'}
        self.plan['tasks'].append(selection_task)
        selection_folder = Path(self.plan['output']) / 'selected'
        selection_folder.mkdir()
        artifacts = {}
        for budget, key in dependencies.items():
            training_folder = Path(self.plan['output']) / key
            metadata = json.loads((training_folder / 'operator.json').read_text())
            arrays = dict(metadata['arrays'])
            arrays['path'] = str(training_folder / arrays['path'])
            artifacts[budget] = dict(arrays, operator_identity=metadata['operator_identity'],
                                     task_key=key, budget=int(budget), registry_sha256='d' * 64)
        tuned_config = dict(config, feature_rank=2, ridge=.3, noise_strength=0., noise_type='none')
        tuned_task = {'key': 'tuned-train-20', 'family': 'operator_train', 'method': 'operator_no_noise_tuned',
                      'role': 'meta_train', 'seed': self.seed, 'budgets': [20],
                      'method_config': tuned_config, 'reference_dependency': 'reference', 'requires': ['reference']}
        self.plan['tasks'].append(tuned_task)
        tuned_folder = Path(self.plan['output']) / tuned_task['key']
        tuned_folder.mkdir()
        worker.operator_train_task(self.plan, tuned_task, tuned_folder, source, geometry,
                                   train, identity, 'd' * 64)
        seal_result(self.plan, tuned_task)
        tuned_metadata = json.loads((tuned_folder / 'operator.json').read_text())
        tuned_asset = dict(tuned_metadata['arrays'], path=str(tuned_folder / 'operator.npz'),
            operator_identity=tuned_metadata['operator_identity'], budget=20, registry_sha256='d' * 64)
        selection_path = self.put('r11-test-output/selected/selection.json', {
            'selected_configuration': result['resolved_configuration'], 'selected_candidate_id': 'fixture',
            'status': 'VALIDATED_SELECTION', 'selected_operator_artifacts': {str(self.seed): artifacts},
            'selected_by_method': {'operator_no_noise_tuned': {'configuration': tuned_metadata['configuration'],
                'candidate_id': 'tuned-fixture', 'operator_artifacts': {str(self.seed): {'20': tuned_asset}}}}})
        worker._write_result(self.plan, selection_task, selection_folder, [selection_path], [selection_path])
        seal_result(self.plan, selection_task)
        dynamic_task = {'seed': self.seed, 'method': 'operator',
                        'method_config': {'selection_dependency': 'selected', 'variant': 'operator'}}
        learned = worker._trained_operator(self.plan, dynamic_task, 20, identity, 'd' * 64)
        self.assertEqual(learned.config.feature_rank, 4)
        dynamic_task['method_config']['variant'] = 'operator_no_noise'
        resolved, variant, _ = worker._resolved_operator_config(self.plan, dynamic_task)
        self.assertEqual(variant, 'operator_no_noise')
        self.assertEqual(resolved.noise_type, 'none')
        self.assertEqual(resolved.noise_strength, 0.)
        dynamic_task['method_config']['variant'] = 'operator_no_noise_tuned'
        resolved, _, _ = worker._resolved_operator_config(self.plan, dynamic_task)
        self.assertEqual(resolved.feature_rank, 2)
        self.assertEqual(resolved.ridge, .3)
        tuned_learned = worker._trained_operator(self.plan, dynamic_task, 20, identity, 'd' * 64)
        self.assertEqual(tuned_learned.operator_identity, tuned_metadata['operator_identity'])

    def test_shuffled_training_future_uses_source_prefix_ea(self):
        source, geometry, identity = load_source(self.plan, self.registry, self.seed, 'cpu')
        catalog = open_catalog(self.plan, self.registry, 'meta_train')
        pairs = list(training_pairs(source, geometry, catalog, budgets=[8], device='cpu',
            cache_root=self.root / 'shuffled-feature-cache', source_identity=identity,
            shuffle_futures=True, shuffle_seed=17))
        self.assertEqual(len(pairs), 4)
        for pair in pairs:
            self.assertNotEqual((pair['participant_id'], pair['session_id']),
                                (pair['future_participant_id'], pair['future_session_id']))
            source_session = next(item for item in catalog.plans if
                (item.participant_id, item.session_id) == (pair['participant_id'], pair['session_id']))
            future_session = next(item for item in catalog.plans if
                (item.participant_id, item.session_id) == (pair['future_participant_id'], pair['future_session_id']))
            support = SessionAccess(catalog, source_session, 'cpu').support(8)
            query = SessionAccess(catalog, future_session, 'cpu').query_inputs()
            ea, _ = native('paper3r7.baseline').fit_reference(source, geometry, support, 'support_EA')
            expected_features, expected_logits = worker.feature_batch(source, geometry, query.x, ea)
            np.testing.assert_array_equal(pair['future_features'], expected_features)
            np.testing.assert_array_equal(pair['future_logits'], expected_logits)

    def test_persistent_task_source_reuse_survives_error_without_target_state_reuse(self):
        """Exercise run_task sequencing; CUDA calls are mocked, native science runs on CPU."""
        loader = native('paper3r7.data')
        actual_loader = loader.load_source
        actual_evaluate = worker.evaluate_task
        loaded = []
        snapshots = []

        def cpu_loader(root, seed, requested_device):
            source, geometry, identity = actual_loader(root, seed, 'cpu')
            loaded.append(source)
            snapshots.append({name: value.clone() for name, value in source.state_dict().items()})
            return source, geometry, identity

        def cpu_evaluate(plan, task, folder, source, geometry, catalog, identity, registry_sha, device, registry):
            if task['key'] == 'middle-error':
                raise RuntimeError('recoverable fixture task error')
            return actual_evaluate(plan, task, folder, source, geometry, catalog, identity, registry_sha,
                                   'cpu', registry)

        for key, cohort in (('first-head', 'holdout'), ('middle-error', 'holdout'),
                            ('next-person-head', 'meta_validation')):
            self.plan['tasks'].append({'key': key, 'family': 'evaluate', 'method': 'head_full',
                'method_config': {}, 'seed': self.seed, 'role': cohort, 'cohort': cohort, 'budgets': [20]})
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            with patch.object(loader, 'load_source', side_effect=cpu_loader) as source_loader, \
                    patch.object(worker, 'load_registry', return_value=(self.registry, 'a' * 64)), \
                    patch.object(worker, 'runtime_observations', return_value={'synthetic_fixture': True}), \
                    patch.object(worker, 'validate_gpu_environment'), \
                    patch.object(worker, 'evaluate_task', side_effect=cpu_evaluate), \
                    patch('torch.cuda.is_available', return_value=True), \
                    patch('torch.cuda.set_device'), patch('torch.cuda.manual_seed_all'), \
                    patch('torch.cuda.get_rng_state', side_effect=lambda *_args: torch.get_rng_state()), \
                    patch('torch.cuda.set_rng_state'):
                first = worker.run_task(self.plan, self.plan['tasks'][0], 'cuda:0')
                self.assertEqual(first['status'], 'COMPLETE')
                with self.assertRaisesRegex(RuntimeError, 'recoverable fixture task error'):
                    worker.run_task(self.plan, self.plan['tasks'][1], 'cuda:0')
                third = worker.run_task(self.plan, self.plan['tasks'][2], 'cuda:0')
                self.assertEqual(third['status'], 'COMPLETE')
                self.assertEqual(source_loader.call_count, 1)
                self.assertEqual(worker.run_task(self.plan, self.plan['tasks'][0], 'cuda:0')['status'], 'COMPLETE')
                self.assertEqual(source_loader.call_count, 1)
        finally:
            torch.use_deterministic_algorithms(old_deterministic)
        self.assertEqual(len(loaded), 1)
        self.assertTrue(all(torch.equal(snapshots[0][name], value)
                            for name, value in loaded[0].state_dict().items()))
        first_uids = set()
        third_uids = set()
        for key, destination in (('first-head', first_uids), ('next-person-head', third_uids)):
            folder = Path(self.plan['output']) / key
            for path in folder.glob('sessions/*.prediction.json'):
                destination.update(json.loads(path.read_text())['valid_support_event_uids'])
        self.assertTrue(first_uids and third_uids)
        self.assertFalse(first_uids & third_uids)

    def test_cached_source_detects_in_memory_mutation(self):
        source, _, _ = load_source(self.plan, self.registry, self.seed, 'cpu')
        with torch.no_grad():
            next(source.parameters()).add_(1.)
        with self.assertRaisesRegex(ValueError, 'mutated within the persistent worker'):
            load_source(self.plan, self.registry, self.seed, 'cpu')
        restored, _, _ = load_source(self.plan, self.registry, self.seed, 'cpu')
        self.assertIsNot(restored, source)

    def test_source_cache_miss_and_hit_preserve_identical_task_rng(self):
        torch.manual_seed(8021)
        state = torch.get_rng_state().clone()
        first, _, _ = load_source(self.plan, self.registry, self.seed, 'cpu')
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        miss_sequence = torch.rand(16)
        torch.set_rng_state(state)
        second, _, _ = load_source(self.plan, self.registry, self.seed, 'cpu')
        self.assertIs(first, second)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        torch.testing.assert_close(torch.rand(16), miss_sequence, rtol=0, atol=0)

    def test_prepared_campaign_native_train_selection_and_target_paths(self):
        registry_path = self.put('runs/round11_preparation/study-registry.json', self.registry)
        plan = study.build_plan(self.root, describe(registry_path, self.root), stage='campaign',
            grid={'feature_ranks': [4], 'noise_strengths': [1.], 'ridges': [.01]},
            seeds=[self.seed], cohorts=['holdout'])
        plan['synthetic_fixture'] = True
        plan = sign_plan(plan)
        study.write_plan(plan, Path(plan['output']) / 'plan.json')
        source, geometry, identity = load_source(plan, self.registry, self.seed, 'cpu')
        registry_sha = plan['study_registry']['sha256']

        def execute(task):
            folder = Path(plan['output']) / task['key']
            folder.mkdir(parents=True, exist_ok=True)
            if task['family'] == 'selection':
                result = worker.run_task(plan, task, 'cpu')
            else:
                catalog = open_catalog(plan, self.registry, task['cohort'])
                if task['family'] == 'feature_reference':
                    result = worker.feature_reference_task(plan, task, folder, source, geometry,
                        catalog, identity, registry_sha, self.registry)
                elif task['family'] == 'operator_train':
                    result = worker.operator_train_task(plan, task, folder, source, geometry,
                        catalog, identity, registry_sha)
                else:
                    result = worker.evaluate_task(plan, task, folder, source, geometry,
                        catalog, identity, registry_sha, 'cpu', self.registry)
            seal_result(plan, task)
            return result

        for task in plan['tasks']:
            if task['phase'] == 'development':
                execute(task)
        selected_task = next(task for task in plan['tasks'] if task['family'] == 'selection')
        decision = json.loads((Path(plan['output']) / selected_task['key'] / 'selection.json').read_text())
        self.assertEqual(set(decision['selected_by_method']),
                         {'operator', 'operator_no_noise_tuned', 'operator_erm_tuned'})
        self.assertEqual(decision['status'], 'VALIDATED_SELECTION')
        self.assertFalse(decision['target_query_labels_used_for_selection'])
        for task in plan['tasks']:
            if task['phase'] != 'development' and task['method'] in (
                    'operator', 'operator_no_noise_tuned', 'operator_erm_tuned', 'operator_symmetric'):
                result = execute(task)
                self.assertEqual(result['status'], 'COMPLETE')
                if task['family'] == 'evaluate':
                    self.assertIn('resolved_configuration', result)

    def test_catalog_metadata_is_cached_once_per_role_and_frozen(self):
        reader = native('paper3r7.data')
        actual_catalog = reader.Catalog
        with patch.object(reader, 'Catalog', wraps=actual_catalog) as constructor:
            training = open_catalog(self.plan, self.registry, 'meta_train')
            validation = open_catalog(self.plan, self.registry, 'meta_validation')
            for _ in range(3):
                self.assertIs(open_catalog(self.plan, self.registry, 'meta_train'), training)
                self.assertIs(open_catalog(self.plan, self.registry, 'meta_validation'), validation)
            self.assertEqual(constructor.call_count, 2)
            with patch('paper3r11.data.input_artifact', side_effect=AssertionError('Cache hit must not rehash upstream')):
                self.assertIs(open_catalog(self.plan, self.registry, 'meta_train'), training)
            self.assertEqual(training.cache_limit, 0)
            with self.assertRaises(TypeError):
                training.role = 'internal_holdout'
            with self.assertRaises(TypeError):
                training.plans[0].rows[next(iter(training.plans[0].rows))]['label'] = 1
            train_access = SessionAccess(training, training.plans[0], 'cpu')
            val_access = SessionAccess(validation, validation.plans[0], 'cpu')
            self.assertNotEqual(train_access.identity, val_access.identity)
            with self.assertRaisesRegex(ValueError, 'prediction SHA256 seal'):
                val_access.evaluation_labels(self.root / 'absent.npz', '0' * 64)
            with self.assertRaisesRegex(ValueError, 'meta_train participants'):
                val_access.meta_train_future_labels()
            clear_source_cache()
            self.assertIsNot(open_catalog(self.plan, self.registry, 'meta_train'), training)
            self.assertEqual(constructor.call_count, 3)


if __name__ == '__main__':
    unittest.main()
