"""Finite CPU fixtures: capability separation, exact recovery and protocol identity.

These are synthetic software tests. No clinical dataset or GPU is loaded.
"""
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from paper3r10 import worker
from paper3r10.data import (SessionAccess, atomic_npz, cached_feature_batch, describe,
                           input_artifact, native, open_catalog, selected_sessions, weighted_reference)
from paper3r10.metrics import participant_summaries, session_metrics
from paper3r10.platform.common import digest, read, seal_result, sha, verify_result


class GuardedRow(dict):
    def __init__(self, *args, query=False, allow=None, **kwargs):
        super().__init__(*args, **kwargs); self.query = query; self.allow = allow or (lambda: False)
    def __getitem__(self, key):
        if key == 'label' and self.query and not self.allow():
            raise AssertionError('Future labels opened before persisted predictions')
        return super().__getitem__(key)


class FixtureCatalog:
    def __init__(self, count=1, role='internal_holdout', allow=None):
        self.role = role; self.signal_sha = 'f' * 64
        self.identity = {'manifest_sha256': '1' * 64, 'split_sha256': '2' * 64,
                         'signal_contract_sha256': self.signal_sha}
        self.order_scope = 'synthetic_fixture_verified_order'; self.plans = []
        self.values = {}; self.materialize_calls = 0
        rng = np.random.default_rng(20261018)
        for person in range(count):
            participant = 'P' + str(person + 1)
            rows = {}
            acquired = tuple(f'{participant}-e{i}' for i in range(20))
            query = tuple(f'{participant}-e{i}' for i in range(20, 26))
            for i, uid in enumerate(acquired + query):
                rows[uid] = GuardedRow(event_uid=uid, cache_index=i, label=i % 2,
                    valid=i not in (2, 5), physical_run_id='r1', query=i >= 20, allow=allow)
            session = SimpleNamespace(participant_id=participant, session_id='1', rows=rows,
                audit={'acquired_prefix_uids': acquired, 'query_uids': query})
            self.plans.append(session)
            values = rng.normal(size=(26, 11, 4)); values -= values.mean(1, keepdims=True)
            self.values[participant] = values
    def array(self, session): return self.values[session.participant_id]
    def materialize(self, *args, **kwargs):
        self.materialize_calls += 1
        raise AssertionError('Unsafe native materialize exposes query labels')


class DataWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads(); torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.root = Path(self.temporary.name)
        torch.manual_seed(17)
        geometry = native('p3work.spatial_geometry')
        self.geometry = geometry.SpatialGeometry.from_reference(torch.eye(10, dtype=torch.float64) / 10)
        adapt = native('p3work.adapt')
        self.source = adapt.FrozenSourceModel(nn.Sequential(nn.Flatten(), nn.Linear(44, 6), nn.Tanh()).double(),
                                             nn.Linear(6, 2).double(), feature_dim=6)
        self.source_identity = {'source_checkpoint_sha256': 'c' * 64,
                                'signal_contract_sha256': 'f' * 64}
        self.reference_task = {'key': 'reference', 'family': 'feature_reference', 'role': 'meta_train', 'seed': 20261018}
        self.task = {'key': 'candidate', 'family': 'evaluate', 'role': 'internal_holdout',
                     'cohort': 'holdout', 'seed': 20261018, 'method': 'profile_map',
                     'reference_dependency': 'reference', 'requires': ['reference'], 'budgets': [4],
                     'method_config': {'amplitude': .1, 'max_newton': 12}}
        self.plan = {'root': str(self.root), 'output': str(self.root / 'r10-results'),
                     'synthetic_fixture': True,
                     'stage': 'formal_exploratory', 'recipe_sha256': 'a' * 64,
                     'tasks': [self.reference_task, self.task], 'reference_strategy': 'build_meta_train_prefix'}
        self.ref_folder = Path(self.plan['output']) / 'reference'; self.ref_folder.mkdir(parents=True)
        self.folder = Path(self.plan['output']) / 'candidate'; self.folder.mkdir()
        worker._STOP_REQUESTED.clear()

    def tearDown(self):
        worker._STOP_REQUESTED.clear(); self.temporary.cleanup()

    def build_reference(self):
        train = FixtureCatalog(role='meta_train', allow=lambda: False)
        worker.feature_reference_task(self.plan, self.reference_task, self.ref_folder,
                                      self.source, self.geometry, train, self.source_identity, 'd' * 64)
        seal_result(self.plan, self.reference_task)
        self.assertEqual(train.materialize_calls, 0)
        return train

    def catalog(self):
        return FixtureCatalog(allow=lambda: any(self.folder.glob('sessions/*.predictions.npz')))

    def evaluate(self, catalog=None):
        return worker.evaluate_task(self.plan, self.task, self.folder, self.source, self.geometry,
                                    catalog or self.catalog(), self.source_identity, 'd' * 64, 'cpu')

    def test_original_positions_paid_and_future_query_fixed(self):
        catalog = FixtureCatalog(); access = SessionAccess(catalog, catalog.plans[0], 'cpu')
        support4 = access.support(4); support20 = access.support(20)
        self.assertEqual(len(support4.acquired_event_uids), 4)
        self.assertEqual(len(support4.event_uids), 3)
        self.assertEqual(len(support20.acquired_event_uids), 20)
        self.assertEqual(len(support20.event_uids), 18)
        before = access.query_inputs(); after = access.query_inputs()
        self.assertEqual(before.event_uids, after.event_uids)
        self.assertFalse(hasattr(before, 'y')); self.assertFalse(hasattr(before, 'query_y'))
        with self.assertRaises(ValueError): access.evaluation_labels(self.root / 'missing.npz', '0' * 64)
        self.assertEqual(catalog.materialize_calls, 0)

    def test_label_capability_checks_sha_and_uid_seal(self):
        catalog = FixtureCatalog(allow=lambda: True); access = SessionAccess(catalog, catalog.plans[0], 'cpu')
        path = self.root / 'predictions.npz'
        atomic_npz(path, probabilities=np.full((6, 2), .5), event_uids=np.asarray(access.query_ids),
                   session_identity=np.asarray(access.identity))
        np.testing.assert_array_equal(access.evaluation_labels(path, sha(path)), [0, 1, 0, 1, 0, 1])
        with self.assertRaises(ValueError): access.evaluation_labels(path, '0' * 64)
        atomic_npz(path, probabilities=np.full((6, 2), .5), event_uids=np.asarray(access.query_ids[::-1]),
                   session_identity=np.asarray(access.identity))
        with self.assertRaises(ValueError): access.evaluation_labels(path, sha(path))

    def test_reference_has_no_query_reads_and_exact_recovery(self):
        self.build_reference(); catalog = self.catalog()
        result = self.evaluate(catalog)
        self.assertEqual(result['status'], 'COMPLETE'); self.assertEqual(catalog.materialize_calls, 0)
        self.assertTrue(verify_result(self.plan, self.task, sealed=False))
        seal_result(self.plan, self.task); self.assertTrue(verify_result(self.plan, self.task))
        original = (self.folder / 'session_metrics.csv').read_bytes()
        with patch('paper3r10.worker.algorithm.fit', side_effect=AssertionError('Completed fit must be reused')):
            self.evaluate(catalog)
        self.assertEqual((self.folder / 'session_metrics.csv').read_bytes(), original)
        with np.load(next(self.folder.glob('sessions/*.predictions.npz')), allow_pickle=False) as prediction:
            self.assertNotIn('labels', prediction.files)
            self.assertEqual(tuple(prediction['event_uids']), tuple(catalog.plans[0].audit['query_uids']))
        state_path = next(self.folder.glob('sessions/*.fit.npz'))
        self.assertEqual(worker.algorithm.load_state(state_path).config.mode, 'map')
        row = read(next(self.folder.glob('sessions/*.metrics.json')))
        self.assertEqual(row['query_signature'], digest(catalog.plans[0].audit['query_uids']))
        self.assertIn('anchor_ea_ba', row); self.assertIn('incremental_prediction_seconds', row)

    def test_prediction_survives_metrics_failure_without_refit(self):
        self.build_reference(); catalog = self.catalog()
        with patch('paper3r10.worker.session_metrics', side_effect=RuntimeError('Simulated metric interruption')):
            with self.assertRaises(RuntimeError): self.evaluate(catalog)
        self.assertEqual(len(list(self.folder.glob('sessions/*.predictions.npz'))), 1)
        self.assertFalse(list(self.folder.glob('sessions/*.complete.json')))
        with patch('paper3r10.worker.algorithm.fit', side_effect=AssertionError('Persisted prediction must be reused')):
            self.evaluate(catalog)
        self.assertTrue(verify_result(self.plan, self.task, sealed=False))

    def test_true_ea_anchor_matches_native_for_each_query(self):
        self.task['method'] = 'support_ea'; self.task['method_config'] = {}
        catalog = self.catalog(); self.evaluate(catalog)
        with np.load(next(self.folder.glob('sessions/*.predictions.npz')), allow_pickle=False) as saved:
            np.testing.assert_allclose(saved['probabilities'], saved['ea_probabilities'], atol=2e-16, rtol=0)
        self.assertTrue(list(self.folder.glob('sessions/*.fit.npz')))

    def test_metadata_only_subset_balances_people_and_is_fixed(self):
        catalog = FixtureCatalog(count=12)
        subset = {'max_participants': 8, 'max_sessions': 8, 'max_sessions_per_participant': 1,
                  'salt': 'r10-mechanism-v1'}
        first = selected_sessions(catalog, subset); second = selected_sessions(catalog, subset)
        self.assertEqual([s.participant_id for s in first], [s.participant_id for s in second])
        self.assertEqual(len({s.participant_id for s in first}), 8)
        self.assertFalse(any('query_y' in inspect.signature(x).parameters for x in
                             (worker.algorithm.fit, worker.algorithm.predict)))

    def test_fixed_eight_participant_selection_check_rows(self):
        self.plan.update(stage='mechanism', session_subset={'max_participants': 8, 'max_sessions': 8,
            'max_sessions_per_participant': 1, 'salt': 'r10-mechanism-v1'})
        self.task.update(method='support_ea', method_config={}, role='meta_validation', cohort='meta_validation')
        catalog = FixtureCatalog(count=12, role='meta_validation',
                                 allow=lambda: any(self.folder.glob('sessions/*.predictions.npz')))
        self.evaluate(catalog)
        import csv
        rows = list(csv.DictReader((self.folder / 'session_metrics.csv').open()))
        self.assertEqual(len(rows), 8)
        self.assertEqual(sum(x['selection_split'] == 'selection' for x in rows), 4)
        self.assertEqual(sum(x['selection_split'] == 'check' for x in rows), 4)
        self.assertTrue(all(float(x['updated_query_fraction']) == 0. for x in rows))

    def test_label_free_feature_cache_identity_sha_and_recovery(self):
        catalog = FixtureCatalog(); access = SessionAccess(catalog, catalog.plans[0], 'cpu')
        support = access.support(4)
        calls = []
        original = self.source.features_and_logits
        def tracked(x): calls.append(len(x)); return original(x)
        with patch.object(self.source, 'features_and_logits', side_effect=tracked):
            first = cached_feature_batch(self.root / 'cache', self.source, self.geometry,
                support.x, support.event_uids, {'source': self.source_identity, 'data': catalog.identity})
            second = cached_feature_batch(self.root / 'cache', self.source, self.geometry,
                support.x, support.event_uids, {'source': self.source_identity, 'data': catalog.identity})
            self.assertEqual(len(calls), 1)
            np.testing.assert_array_equal(first[0], second[0])
            path = next((self.root / 'cache').glob('*/*.npz')); path.write_bytes(b'corrupt')
            third = cached_feature_batch(self.root / 'cache', self.source, self.geometry,
                support.x, support.event_uids, {'source': self.source_identity, 'data': catalog.identity})
            self.assertEqual(len(calls), 2); np.testing.assert_array_equal(first[0], third[0])
        for path in (self.root / 'cache').glob('*/*.npz'):
            with np.load(path, allow_pickle=False) as saved:
                self.assertNotIn('labels', saved.files); self.assertNotIn('query_y', saved.files)

    def test_safe_native_tensor_extraction_and_same_source_proof(self):
        path = self.root / 'basis.pt'; torch.save({'b': torch.eye(99, 4)}, path)
        descriptor = dict(describe(path, self.root), field=['b'])
        proof = {'seed': 20261018, 'source_identity': self.source_identity,
                 'target_query_labels_used': False, 'training_role': 'meta_train'}
        registry = {'baseline_artifacts': {'20261018': {'sgd10': {
            'spatial_basis': descriptor, 'basis_provenance': proof}}}}
        result = worker._artifact_references(self.plan, registry, 'sgd10', 20261018,
                                            'cpu', self.source_identity)
        self.assertEqual(result['spatial_basis'].shape, (99, 4))
        proof['source_identity'] = {'source_checkpoint_sha256': 'wrong'}
        with self.assertRaisesRegex(ValueError, 'another frozen source'):
            worker._artifact_references(self.plan, registry, 'sgd10', 20261018,
                                        'cpu', self.source_identity)

    def test_reuses_frozen_r9_prefix_reference_without_opening_eeg_or_queries(self):
        catalog = FixtureCatalog(role='meta_train', allow=lambda: False)
        proof = {'role': 'meta_train_prefix20_only', 'preprocessing': 'ea',
                 'data_identity': catalog.identity, 'future_queries_read': False}
        checkpoint = self.root / 'r9-selected-reference.pt'
        torch.save({'identity': 'frozen-native-r9-identity', 'reference': {
            'mean': torch.zeros(6), 'whitener': torch.eye(6), 'lengthscale': 1.,
            'provenance': proof, 'diagnostics': {}}}, checkpoint)
        result_path = self.root / 'r9-reference-result.json'
        result_path.write_text(json.dumps({'status': 'COMPLETE',
            'task': {'family': 'feature_reference', 'seed': 20261018},
            'source_identity': self.source_identity, 'data_identity': catalog.identity,
            'holdout_used_for_selection': False, 'identity': 'frozen-native-r9-identity',
            'selected_checkpoint': {'sha256': sha(checkpoint)}}))
        registry = {'profile_reference_by_seed': {'20261018': {
            'checkpoint': describe(checkpoint, self.root), 'result': describe(result_path, self.root)}}}
        self.plan['reference_strategy'] = 'reuse_r9_ea'
        with patch.object(catalog, 'array', side_effect=AssertionError('No EEG read while reusing reference')):
            worker.feature_reference_task(self.plan, self.reference_task, self.ref_folder,
                self.source, self.geometry, catalog, self.source_identity, 'd' * 64, registry)
        metadata = read(self.ref_folder / 'reference.json')
        self.assertEqual(metadata['reference_strategy'], 'reuse_r9_ea')
        self.assertEqual(metadata['references']['ea']['provenance']['role'], 'meta_train')
        self.assertEqual(metadata['references']['ea']['provenance']['original_provenance'], proof)

    def test_strong_spatial_control_state_is_pickle_free_and_keeps_dtype(self):
        from paper3r10.baselines import fit_baseline
        catalog = FixtureCatalog(); support = SessionAccess(catalog, catalog.plans[0], 'cpu').support(20)
        basis = torch.linalg.qr(torch.randn(99, 4, dtype=torch.float64)).Q
        for method in ('sgd10', 'r3_protected', 'r4_xfit_select_stack'):
            fitted = fit_baseline(method, self.source, self.geometry, support,
                {'spatial_basis': basis, 'basis_provenance': {'training_role': 'meta_train'}},
                {'tangent_steps': 2})
            path = self.root / (method + '.npz'); worker._save_baseline_state(path, fitted)
            with np.load(path, allow_pickle=False) as saved:
                metadata = json.loads(str(saved['metadata'].item()))
                self.assertFalse(metadata['source_model_included'])
                self.assertTrue(any(saved[key].dtype == np.float64 for key in saved.files if key != 'metadata'))

    def test_independent_cache_requires_review_disjointness_and_complete_runs(self):
        contracts = self.root / 'contracts'; contracts.mkdir()
        contract = {'n_times': 128, 'synthetic_fixture': True}
        (contracts / 'input_contract.json').write_text(json.dumps(contract))
        x_path = self.root / 'new.npy'; np.save(x_path, np.zeros((40, 11, 128)))
        rows = [{'participant_id': 'NEW:P1', 'session_id': '1', 'tasknumber': 1,
                 'event_uid': f'new{i}', 'trial_uid': f'new{i}', 'original_trial_index': i,
                 'label': i % 2, 'valid': True, 'cache_index': i,
                 'physical_run_id': 'prefix' if i < 20 else 'future',
                 'physical_run_complete_verified': True} for i in range(40)]
        events = self.root / 'new-events.json'; events.write_text(json.dumps(rows))
        review = self.root / 'review.json'
        review.write_text(json.dumps({'status': 'REVIEWED', 'prior_design_history_reviewed': True,
            'previously_used_for_algorithm_design': False, 'reviewer': 'synthetic-software-fixture',
            'reviewed_utc': '2026-10-26T00:00:00Z', 'dataset_identity': {'id': 'fixture'},
            'participants': ['NEW:P1']}))
        manifest = {'schema_version': 'paper3-r10-independent-cache-v1', 'status': 'COMPLETE',
            'cohort_kind': 'independent_confirmation', 'acquisition_order_verified': True,
            'physical_run_id_available': True, 'previously_used_for_algorithm_design': False,
            'signal_contract_sha256': digest(contract), 'dataset_identity': {'id': 'fixture'},
            'participants': ['NEW:P1'], 'independence_review': describe(review, self.root),
            'sessions': [{'participant_id': 'NEW:P1', 'session_id': '1', 'acquisition_order_verified': True,
                          'X': describe(x_path, self.root), 'events': describe(events, self.root)}]}
        manifest_path = self.root / 'new-manifest.json'; manifest_path.write_text(json.dumps(manifest))
        self.plan['confirmation_manifest'] = describe(manifest_path, self.root)
        registry = {'previously_used_participants': ['OLD:P1']}
        catalog = open_catalog(self.plan, registry, 'confirmation')
        self.assertEqual(catalog.array(catalog.plans[0]).shape, (40, 11, 128))
        access = SessionAccess(catalog, catalog.plans[0], 'cpu')
        self.assertEqual(len(access.query_inputs().event_uids), 20)
        registry['previously_used_participants'] = ['NEW:P1']
        with self.assertRaisesRegex(ValueError, 'previously used'):
            open_catalog(self.plan, registry, 'confirmation')

    def test_participant_unit_and_balanced_metrics(self):
        y = np.asarray([0, 0, 0, 1]); p = np.asarray([[.8, .2], [.8, .2], [.8, .2], [.3, .7]])
        result = session_metrics(y, p, anchors={'ea': np.full((4, 2), .5)})
        self.assertEqual(result['ba'], 1.)
        self.assertAlmostEqual(result['balanced_brier'], .5 * (.2 ** 2 + .3 ** 2))
        rows = [{'participant_id': 'A', 'seed': 1, 'budget': 4, 'ba': x} for x in (.2, .4)]
        rows += [{'participant_id': 'B', 'seed': 1, 'budget': 4, 'ba': .8}]
        output = participant_summaries(rows)
        self.assertAlmostEqual(output[0]['ba'], .3); self.assertEqual(output[0]['n_sessions'], 2)

    def test_reference_degenerate_finite_and_absolute_input_descriptor(self):
        ref = weighted_reference(np.ones((4, 3)), np.ones(4), {'role': 'meta_train'})
        self.assertTrue(np.isfinite(ref['whitener']).all()); self.assertEqual(ref['lengthscale'], 1.)
        path = self.root / 'external.json'; path.write_text('{}')
        descriptor = {'path': str(path), 'bytes': 2, 'sha256': sha(path)}
        self.assertEqual(input_artifact(self.root / 'another-root', descriptor), path)
        path.write_text('{ }')
        with self.assertRaises(ValueError): input_artifact(self.root, descriptor)

    def test_cpu_real_data_is_blocked_and_stop_preserves_completed_units(self):
        with self.assertRaisesRegex(ValueError, 'allocated CUDA'):
            worker.run_task(self.plan, self.task, 'cpu')
        self.build_reference(); worker._STOP_REQUESTED.set()
        with self.assertRaises(worker.PauseRequested): self.evaluate()
        self.assertTrue(verify_result(self.plan, self.reference_task))
        self.assertFalse((self.folder / 'result.json').exists())

    def test_runtime_is_observation_and_joined_to_both_archive_channels(self):
        props = SimpleNamespace(uuid='software-fixture', name='synthetic-device', total_memory=4096,
                                major=8, minor=0)
        with patch('torch.cuda.get_device_properties', return_value=props):
            runtime = worker.runtime_observations(self.source, 'cuda:0')
        self.assertEqual(runtime['device_uuid'], 'software-fixture')
        self.assertEqual(runtime['numpy'], np.__version__)
        self.assertTrue(runtime['hardware_identity_is_observation_not_scientific_source_identity'])
        from paper3r10.platform.common import atomic
        atomic(self.folder / 'runtime.json', runtime)
        self.task.update(method='support_ea', method_config={})
        result = self.evaluate()
        self.assertIn('runtime.json', result['paper_artifacts'])
        self.assertIn('runtime.json', [x['path'] for x in result['artifacts']])


if __name__ == '__main__': unittest.main()
