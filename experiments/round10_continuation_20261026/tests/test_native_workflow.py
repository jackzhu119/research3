"""Native R10 artifact-format and persistent-worker regression fixtures.

All weights/signals are generated software fixtures, not real EEG results.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from paper3r10 import worker
from paper3r10.data import SessionAccess, describe, load_source, native, open_catalog, clear_source_cache
from paper3r10.platform.common import digest, seal_result
from paper3r10.registry import build_registry

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
        self.plan = {'root': str(self.root), 'output': str(self.root / 'r10-test-output'),
                     'stage': 'formal_exploratory', 'synthetic_fixture': True, 'recipe_sha256': 'c' * 64,
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

    def historical_artifacts(self, seed, source, geometry, source_identity):
        """Generate native-format training artifacts and exact fixture receipts."""
        train = open_catalog(self.plan, self.registry, 'meta_train')
        gp = native('paper3r9.gp')
        transport = native('paper3r8.transport')

        def checkpoint(name, value, field, role='meta_train'):
            identity = f'fixture-native-{seed}-{name}'
            path = self.root / f'fixtures/{seed}-{name}.pt'
            torch.save(dict(value, identity=identity), path)
            item = dict(describe(path, self.root), field=[field])
            task_key = f'native-{name}-{seed}'
            receipt = self.put(f'fixtures/{seed}-{name}-receipt.json', {
                'status': 'COMPLETE', 'task': {'key': task_key, 'seed': seed},
                'identity': identity, 'source_identity': source_identity, 'data_identity': train.identity,
                'selected_checkpoint': describe(path, self.root)})
            proof = {'seed': seed, 'source_identity': source_identity, 'data_identity': train.identity,
                     'role': role, 'native_identity': identity, 'native_task_key': task_key,
                     'result': describe(receipt, self.root), 'future_queries_read': False,
                     'query_labels_used': False, 'target_query_labels_used': False}
            return item, proof

        generator = torch.Generator().manual_seed(seed)
        basis = torch.linalg.qr(torch.randn(99, 4, dtype=torch.float64, generator=generator)).Q
        basis_item, basis_proof = checkpoint('basis', {'b': basis}, 'b')
        support = SessionAccess(train, train.plans[0], 'cpu').support(20)
        covariance = transport.trial_covariances(geometry.q, support.x)
        contrast = covariance[support.y == 1].mean(0) - covariance[support.y == 0].mean(0)
        contrast = geometry.rref_invsqrt @ contrast @ geometry.rref_invsqrt
        class_item, class_proof = checkpoint('contrast', {'class_reference': contrast}, 'class_reference')
        artifacts = {method: {'spatial_basis': basis_item, 'basis_provenance': basis_proof}
                     for method in ('sgd10', 'sgd100', 'r3_protected', 'r4_xfit_select_stack')}
        artifacts['r8_fiber_bounded'] = {'class_reference': class_item, 'class_reference_provenance': class_proof}
        for mode in ('raw', 'ea'):
            features = []
            for session in train.plans:
                support = SessionAccess(train, session, 'cpu').support(20)
                ea = None
                if mode == 'ea':
                    ea, _ = native('paper3r7.baseline').fit_reference(source, geometry, support, 'support_EA')
                features.append(worker.feature_batch(source, geometry, support.x, ea)[0])
            reference = gp.fit_reference_geometry(torch.from_numpy(np.concatenate(features)),
                provenance={'role': 'meta_train_prefix20_only', 'preprocessing': mode,
                            'future_queries_read': False, 'query_labels_used': False})
            item, proof = checkpoint('reference-' + mode, {'reference': reference.state_dict()}, 'reference')
            config = {'kernel': 'rbf' if mode == 'raw' else 'linear', 'tau': .1,
                      'predictive': 'anchored_laplace'}
            config_proof = dict(proof, role='meta_validation', selection_role='meta_validation')
            artifacts['r9_' + mode + '_anchored'] = {'r9_reference_' + mode: item,
                'r9_reference_provenance_' + mode: proof, 'r9_config_' + mode: config,
                'r9_config_provenance_' + mode: config_proof}
        self.registry['baseline_artifacts'][str(seed)] = artifacts
        return artifacts

    def test_three_seed_formal_all_methods_native_artifact_loading_and_isolated_missing(self):
        from paper3r10.baselines import MissingBaselineArtifact
        methods = ('frozen', 'support_ea', 'head_bias', 'head_rank4', 'head_full', 'sgd10', 'sgd100',
                   'r3_protected', 'r4_xfit_select_stack', 'r8_fiber_bounded', 'r9_raw_anchored',
                   'r9_ea_anchored', 'profile', 'profile_map', 'profile_no_completion',
                   'profile_frozen_anchor', 'profile_quadratic', 'profile_variance_gate')
        self.plan.update(amendment={'path': 'fixture-amendment.json', 'bytes': 2, 'sha256': 'e' * 64},
                         amendment_decision_identity='f' * 64,
                         selection_policy='paired-ea-gain-one-se-exploratory-amendment-v1')
        for seed in (20261018, 20261019, 20261020):
            source, geometry, identity = load_source(self.plan, self.registry, seed, 'cpu')
            before = {name: value.clone() for name, value in source.state_dict().items()}
            self.historical_artifacts(seed, source, geometry, identity)
            reference_task = {'key': f'reference-{seed}', 'family': 'feature_reference',
                              'role': 'meta_train', 'seed': seed}
            self.plan['tasks'].append(reference_task)
            reference_folder = Path(self.plan['output']) / reference_task['key']
            reference_folder.mkdir(parents=True)
            worker.feature_reference_task(self.plan, reference_task, reference_folder, source, geometry,
                open_catalog(self.plan, self.registry, 'meta_train'), identity, 'd' * 64)
            seal_result(self.plan, reference_task)
            catalog = open_catalog(self.plan, self.registry, 'holdout')
            for method in methods:
                task = {'key': f'{method}-{seed}', 'family': 'evaluate', 'method': method,
                        'method_config': {'amplitude': .1, 'max_newton': 12} if method.startswith('profile') else {},
                        'seed': seed, 'role': 'internal_holdout', 'cohort': 'holdout', 'budgets': [20],
                        'reference_dependency': reference_task['key'], 'requires': [reference_task['key']],
                        'session_subset': {'max_participants': 1, 'max_sessions': 1, 'salt': 'native-formal'}}
                self.plan['tasks'].append(task)
                folder = Path(self.plan['output']) / task['key']
                folder.mkdir()
                result = worker.evaluate_task(self.plan, task, folder, source, geometry,
                    catalog, identity, 'd' * 64, 'cpu', self.registry)
                self.assertEqual(result['status'], 'COMPLETE', method)
                self.assertEqual(result['completed_session_budget_cells'], 1, method)
                self.assertEqual(result['amendment_decision_identity'], 'f' * 64)
                seal_result(self.plan, task)
            self.assertTrue(all(torch.equal(before[name], value) for name, value in source.state_dict().items()))
            removed = self.registry['baseline_artifacts'][str(seed)].pop('r8_fiber_bounded')
            with self.assertRaisesRegex(MissingBaselineArtifact, f'method=r8_fiber_bounded, seed={seed}'):
                worker._artifact_references(self.plan, self.registry, 'r8_fiber_bounded', seed, 'cpu', identity)
            self.assertEqual(worker._artifact_references(self.plan, self.registry, 'frozen', seed, 'cpu', identity), {})
            self.registry['baseline_artifacts'][str(seed)]['r8_fiber_bounded'] = removed

    def test_legacy_reference_reuse_copies_exact_bytes_without_source_or_catalog(self):
        import copy
        source, geometry, identity = load_source(self.plan, self.registry, self.seed, 'cpu')
        registry_path = self.put('runs/round10_preparation/study-registry.json', self.registry)
        self.plan['study_registry'] = describe(registry_path, self.root)
        registry_sha = self.plan['study_registry']['sha256']
        old_task = {'key': 'old-reference', 'family': 'feature_reference', 'role': 'meta_train', 'seed': self.seed}
        self.plan['tasks'] = [old_task]
        old_folder = Path(self.plan['output']) / old_task['key']
        old_folder.mkdir(parents=True)
        worker.feature_reference_task(self.plan, old_task, old_folder, source, geometry,
            open_catalog(self.plan, self.registry, 'meta_train'), identity, registry_sha)
        seal_result(self.plan, old_task)
        old_plan_path = self.put('r10-test-output/plan.json', self.plan)
        original_result = json.loads((old_folder / 'result.json').read_text())
        original_files = {name: (old_folder / name).read_bytes() for name in
                          ('result.json', 'completion.json', 'reference.json', 'reference.npz')}
        reuse = {'schema_version': 'paper3-r10-legacy-reference-reuse-v1',
                 'source_recipe_sha256': self.plan['recipe_sha256'], 'task_key': old_task['key'],
                 'task_identity': original_result['task_identity'], 'registry_sha256': registry_sha,
                 'source_identity': identity, 'catalog_identity': original_result['catalog_identity'],
                 'source_plan': describe(old_plan_path, self.root)}
        for name, filename in (('result','result.json'),('completion','completion.json'),
                               ('reference_json','reference.json'),('reference_npz','reference.npz')):
            reuse[name] = describe(old_folder / filename, self.root)
        plan = copy.deepcopy(self.plan)
        plan.update(output=str(self.root / 'r10-continuation'), recipe_sha256='f' * 64)
        task = dict(old_task, key='imported-reference', legacy_reference_reuse=reuse)
        plan['tasks'] = [task]
        with patch.object(worker, 'load_source', side_effect=AssertionError('No source reload')), \
                patch.object(worker, 'open_catalog', side_effect=AssertionError('No metadata scan')), \
                patch.object(worker, 'load_registry', side_effect=AssertionError('No registry rehash')):
            result = worker.run_task(plan, task, 'cpu')
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual(result['recipe_sha256'], 'f' * 64)
        self.assertEqual(result['legacy_reference_reuse']['reuse_kind'], 'EXACT_BYTE_COPY_NO_RETRAINING')
        folder = Path(plan['output']) / task['key']
        for name in ('reference.json', 'reference.npz'):
            self.assertEqual((folder / name).read_bytes(), original_files[name])
        for name, value in original_files.items():
            self.assertEqual((old_folder / name).read_bytes(), value)
        seal_result(plan, task)
        bad = copy.deepcopy(task)
        bad['key'] = 'bad-import'
        bad['legacy_reference_reuse']['reference_npz']['sha256'] = '0' * 64
        plan['tasks'].append(bad)
        with self.assertRaisesRegex(ValueError, 'SHA identity changed'):
            worker.run_task(plan, bad, 'cpu')
        self.assertFalse((Path(plan['output']) / bad['key'] / 'reference.npz').exists())

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
            with patch('paper3r10.data.input_artifact', side_effect=AssertionError('Cache hit must not rehash upstream')):
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
            clear_source_cache()
            self.assertIsNot(open_catalog(self.plan, self.registry, 'meta_train'), training)
            self.assertEqual(constructor.call_count, 3)
