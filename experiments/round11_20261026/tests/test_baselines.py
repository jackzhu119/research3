"""Native-control parity and no-query-fitting tests on a frozen synthetic encoder."""
import copy
import inspect
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from paper3r11.baselines import fit_baseline, predict_baseline, _native, MissingBaselineArtifact


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads=torch.get_num_threads()
        torch.set_num_threads(1)
        cls.native=_native()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(7319)
        m=self.native
        encoder=nn.Sequential(nn.Flatten(),nn.Linear(11*4,6),nn.Tanh()).double()
        self.source=m['adapt'].FrozenSourceModel(encoder,nn.Linear(6,2).double(),feature_dim=6)
        self.geometry=m['geometry'].SpatialGeometry.from_reference(torch.eye(10,dtype=torch.float64)/10)
        x=torch.randn(12,11,4,dtype=torch.float64);x=x-x.mean(1,keepdim=True)
        # Paid acquisition positions include two excluded QC trials.
        acquired=tuple('e'+str(i) for i in range(16))
        events=tuple(acquired[i] for i in (0,1,2,4,5,7,8,9,10,11,13,15))
        self.support=SimpleNamespace(x=x,y=torch.arange(12)%2,event_uids=events,acquired_event_uids=acquired)
        self.query=torch.randn(7,11,4,dtype=torch.float64);self.query-=self.query.mean(1,keepdim=True)
        self.basis=torch.linalg.qr(torch.randn(99,4,dtype=torch.float64)).Q
        self.reference={'spatial_basis':self.basis,'basis_provenance':{'role':'meta_train','seed':20261018}}
        self.source_before=copy.deepcopy(self.source.state_dict())

    def check_frozen(self):
        self.source.assert_unchanged()
        for key,value in self.source_before.items():
            self.assertTrue(torch.equal(value,self.source.state_dict()[key]),key)

    def fit(self,method,reference=None,config=None):
        return fit_baseline(method,self.source,self.geometry,self.support,reference or self.reference,config)

    def test_frozen_exact_and_no_query_fit_argument(self):
        fitted=self.fit('frozen')
        torch.testing.assert_close(fitted.predict_proba(self.query),self.source.source_logits(self.query).softmax(1),rtol=0,atol=2e-16)
        self.assertNotIn('query_x',inspect.signature(fit_baseline).parameters)
        self.assertFalse(fitted.diagnostics['fit_uses_query_labels'])
        self.check_frozen()

    def test_true_ea_native_parity_and_label_independence(self):
        fitted=self.fit('support_ea',config={'variant':'head_newton'})
        state,_=self.native['reference'].fit_reference(self.source,self.geometry,self.support,'support_EA')
        expected=self.native['reference'].reference_logits(self.source,self.geometry,state,self.query).softmax(1)
        torch.testing.assert_close(fitted.predict_proba(self.query),expected,rtol=0,atol=2e-16)
        altered=copy.copy(self.support);altered.y=1-self.support.y
        other=fit_baseline('support_ea',self.source,self.geometry,altered)
        torch.testing.assert_close(fitted.state['transform'],other.state['transform'],rtol=0,atol=0)
        self.assertEqual(fitted.diagnostics['variant'],'support_EA')
        self.assertEqual(fitted.diagnostics['optimizer_updates'],0)
        self.check_frozen()

    def test_head_sgd10_matches_original_full_control(self):
        fitted=self.fit('head_sgd10')
        state,_=self.native['reference'].fit_reference(self.source,self.geometry,self.support,'head_only')
        torch.testing.assert_close(fitted.state['delta_w'],state['delta_w'],rtol=1e-13,atol=1e-15)
        expected=self.native['reference'].reference_logits(self.source,self.geometry,state,self.query)
        torch.testing.assert_close(fitted.predict_proba(self.query),expected.softmax(1),rtol=1e-13,atol=1e-15)
        self.check_frozen()

    def test_low_capacity_heads_have_correct_subspace_and_ridge(self):
        for name,maxrank in [('head_bias',0),('head_rank4',4),('head_full',6)]:
            fitted=self.fit(name)
            self.assertLessEqual(fitted.diagnostics['effective_head_rank'],maxrank)
            self.assertEqual(fitted.diagnostics['head_prior_normalization_dim_including_bias'],7)
            u=fitted.state['head_projector'];w=fitted.state['delta_w']
            torch.testing.assert_close(w,u@(u.T@w),rtol=1e-12,atol=1e-14)
            self.assertEqual(fitted.diagnostics['optimizer_updates'],100)
            self.assertTrue(torch.isfinite(fitted.predict_proba(self.query)).all())
        self.check_frozen()

    def test_missing_spatial_artifact_is_error_not_random_basis(self):
        for name in ('sgd10','sgd100','r3_protected','r4_xfit_select_stack'):
            with self.assertRaises(MissingBaselineArtifact):
                self.fit(name,reference={'basis_provenance':{'role':'meta_train'}})

    def test_actual_sgd_native_parity(self):
        for name in ('sgd10','sgd100'):
            fitted=self.fit(name)
            cfg=self.native['r3'].Config(**fitted.diagnostics['native_config'])
            native=self.native['r3'].fit(self.source,self.geometry,self.basis,self.support,cfg,create_graph=False)
            expected=self.native['r3'].predict(self.source,self.geometry,self.basis,native,self.query)
            torch.testing.assert_close(fitted.predict_proba(self.query),expected.softmax(1),rtol=1e-12,atol=1e-14)
            self.assertTrue(fitted.state_dict()['source_model_included'] is False)
        self.check_frozen()

    def test_r3_protection_native_parity_and_stable_log_score(self):
        fitted=self.fit('r3_protected',config={'tangent_steps':2})
        cfg=self.native['r3'].Config(**fitted.diagnostics['native_config'])
        native=self.native['r3'].fit(self.source,self.geometry,self.basis,self.support,cfg,create_graph=False)
        expected=self.native['r3'].predict(self.source,self.geometry,self.basis,native,self.query)
        fitted.predict_proba(self.query)
        torch.testing.assert_close(fitted.last_log_probabilities,expected.log_softmax(1),rtol=1e-12,atol=1e-14)
        self.assertGreaterEqual(fitted.state.protection_fraction,0)
        self.assertLessEqual(fitted.state.protection_fraction,1)
        fitted.state_dict()
        self.check_frozen()

    def test_r4_native_oof_selection_and_paid_position_folds(self):
        # Keep the prespecified grid: exercise selection/stacking, not an in-sample stand-in.
        fitted=self.fit('r4_xfit_select_stack',config={'tangent_steps':2})
        cfg=self.native['r4'].Config(**fitted.diagnostics['native_config'])
        native=self.native['r4'].fit(self.source,self.geometry,self.basis,self.support,cfg,create_graph=False)
        expected=self.native['r4'].predict(self.source,self.geometry,self.basis,native,self.query)
        torch.testing.assert_close(fitted.predict_proba(self.query),expected.softmax(1),rtol=1e-11,atol=1e-13)
        cv=fitted.diagnostics['cv']
        self.assertEqual(cv['selection_labels'],'acquired_support_only')
        self.assertGreater(cv['n_candidate_fits'],1)
        # Explicit archive conversion must handle nested fitted states with tensors.
        fitted.state_dict()
        self.check_frozen()

    def test_query_partition_does_not_change_ea_or_head_predictions(self):
        for name in ('support_ea','head_rank4'):
            fitted=self.fit(name)
            together=fitted.predict_proba(self.query)
            split=torch.cat((fitted.predict_proba(self.query[:3]),fitted.predict_proba(self.query[3:])))
            torch.testing.assert_close(together,split,rtol=1e-12,atol=1e-14)
            self.assertFalse(fitted.diagnostics['fit_uses_query_inputs'])
        self.check_frozen()

    def test_r8_fiber_native_parity_and_reference_role(self):
        contrast=torch.diag(torch.linspace(-.1,.1,10,dtype=torch.float64))
        ref={'class_reference':contrast,'class_reference_provenance':{'role':'meta_train'}}
        fitted=self.fit('r8_fiber_bounded',ref)
        cfg=self.native['transport'].TransportConfig(**fitted.diagnostics['native_config'])
        native=self.native['transport'].fit_transport(self.source,self.geometry,self.support,cfg,meta_parameters={'class_reference':contrast})
        expected=self.native['transport'].predict_transport(self.source,self.geometry,native,self.query)
        torch.testing.assert_close(fitted.predict_proba(self.query),expected.softmax(1),rtol=1e-12,atol=1e-14)
        ref['class_reference_provenance']={'role':'holdout'}
        with self.assertRaises(ValueError):self.fit('r8_fiber_bounded',ref)
        self.check_frozen()

    def gp_reference(self,mode,kernel=None):
        # Independent synthetic meta_train features; target queries never used.
        f=torch.randn(30,6,dtype=torch.float64)
        ref=self.native['gp'].fit_reference_geometry(f,provenance={'role':'meta_train_prefix20_only','preprocessing':mode})
        return {'r9_reference_'+mode:ref.state_dict(),
                'r9_config_'+mode:{'kernel':kernel or ('rbf' if mode=='raw' else 'linear'),'tau':.3,'predictive':'anchored_laplace'},
                'r9_config_provenance_'+mode:{'role':'meta_validation'}}

    def test_r9_raw_and_true_ea_native_gp_parity(self):
        for mode,name in [('raw','r9_raw_anchored'),('ea','r9_ea_anchored')]:
            ref=self.gp_reference(mode);fitted=self.fit(name,ref)
            x=self.support.x;q=self.query
            if mode=='ea':
                ea,_=self.native['reference'].fit_reference(self.source,self.geometry,self.support,'support_EA')
                x=ea['transform']@x;q=ea['transform']@q
            f,base=self.source.features_and_logits(self.native['geometry'].joint_trial_rms(x))
            qf,qbase=self.source.features_and_logits(self.native['geometry'].joint_trial_rms(q))
            native=self.native['gp'].fit_gp(f,base,self.support.y,
                self.native['gp'].GPConfig(**fitted.diagnostics['native_config']),
                self.native['gp'].ReferenceGeometry.from_state_dict(ref['r9_reference_'+mode]))
            expected=native.predict_proba(qf,qbase)
            torch.testing.assert_close(fitted.predict_proba(self.query),expected,rtol=1e-12,atol=1e-14)
            self.assertIn('state',fitted.state_dict())
        self.check_frozen()

    def test_r9_missing_or_query_selected_reference_rejected(self):
        with self.assertRaises(MissingBaselineArtifact):self.fit('r9_raw_anchored',{})
        ref=self.gp_reference('raw')
        ref['r9_reference_raw']['provenance']['query_labels_used']=True
        with self.assertRaises(ValueError):self.fit('r9_raw_anchored',ref)

    def test_raw_r9_public_representative_requires_rbf_and_preserves_explicit_legacy_linear(self):
        ref=self.gp_reference('raw',kernel='linear')
        with self.assertRaisesRegex(ValueError,'rbf/anchored_laplace'):
            self.fit('r9_raw_anchored',ref)
        old=self.fit('r9_raw_linear_anchored',ref)
        self.assertEqual(old.diagnostics['native_config']['kernel'],'linear')
        current=self.fit('r9_raw_anchored',self.gp_reference('raw'))
        self.assertEqual(current.diagnostics['native_config']['kernel'],'rbf')

    def test_one_class_fallback_preserves_frozen_source(self):
        self.support.y.zero_()
        for name in ('head_rank4','sgd10','r3_protected'):
            fitted=self.fit(name)
            self.assertEqual(fitted.diagnostics['fallback_reason'],'insufficient_class_support')
            torch.testing.assert_close(fitted.predict_proba(self.query),self.source.source_logits(self.query).softmax(1),rtol=0,atol=2e-16)
        self.check_frozen()

    def test_reject_paid_prefix_over_20_and_variant_mismatch(self):
        self.support.acquired_event_uids=tuple(str(i) for i in range(21))
        with self.assertRaises(ValueError):self.fit('frozen')
        self.support.acquired_event_uids=tuple('e'+str(i) for i in range(16))
        with self.assertRaises(ValueError):self.fit('sgd10',config={'variant':'head_newton'})


if __name__=='__main__':unittest.main()
