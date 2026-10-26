"""Bounded historical-artifact discovery never imports Torch or scans raw data."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from paper3r11.registry import (baseline_availability, build_registry, write_registry,
                               validate_registry, describe)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'paper3'; self.root.mkdir()
        self.seed=20261018
        for seed in (20261018,20261019,20261020):
            self.json(self.root/f'runs/source_training_v2/f0_receipt_seed-{seed}.json',{'status':'SOURCE_COMPLETE','seed':seed})
            self.json(self.root/f'runs/training_campaign_v1/references/seed-{seed}/receipt.json',{'status':'REFERENCE_COMPLETE','seed':seed})
        split=self.json(self.root/'data/participant_split.json',{'split_sha256':'b'*64,'partitions':{'meta_train':['S1'],'meta_validation':['S2'],'internal_holdout':['S3']}})
        manifest=self.json(self.root/'data/cache_manifest.json',{'status':'STIEGER_CACHE_COMPLETE','input_contract_sha256':'c'*64,
            'participant_split':{'path':'data/participant_split.json','sha256':hashlib.sha256(split.read_bytes()).hexdigest()},'sessions':[]})
        self.json(self.root/'status/tasks.json',{'W3':{'cache_manifest':{'path':'data/cache_manifest.json','sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}}})

    def tearDown(self): self.temp.cleanup()

    def json(self,path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path

    def make_plan(self,number,tasks,relative='plan.json'):
        dirname=f'followup_20261024_r{number}' if number in(3,4) else f'followup_20261025_r{number}'
        output=self.root/'runs'/dirname
        plan={'round':number,'root':str(self.root),'output':str(output),'recipe_sha256':str(number)*64,'tasks':tasks}
        self.json(output/relative,plan)
        return plan,output

    def complete(self,plan,output,task,filename,**extra):
        folder=output/task['key'];folder.mkdir(parents=True,exist_ok=True)
        checkpoint=folder/filename;checkpoint.write_bytes(b'safe-tensor-file-placeholder-not-loaded-during-discovery')
        receipt={'status':'COMPLETE','task':task,'recipe_sha256':plan['recipe_sha256'],
                 'holdout_used_for_selection':False,'identity':'identity-'+task['key'],
                 'source_identity':{'source_receipt_sha256':'a'*64},'data_identity':{'split_sha256':'b'*64},
                 'selected_checkpoint':{'path':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()},**extra}
        self.json(folder/'result.json',receipt)
        return receipt,checkpoint

    def r9(self):
        tasks=[]
        for mode in('raw','ea'):
            kernel='rbf' if mode=='raw' else 'linear'
            tasks.append({'key':f'feature-reference-{mode}-seed{self.seed}','family':'feature_reference','seed':self.seed,'config':{'preprocessing':mode}})
            tasks.append({'key':f'select-{mode}-{kernel}-seed{self.seed}','family':'select','seed':self.seed,'config':{'preprocessing':mode,'kernel':kernel},'selection_role':'meta_validation'})
        plan,output=self.make_plan(9,tasks)
        for task in tasks:
            mode=task['config']['preprocessing']
            if task['family']=='feature_reference':
                self.complete(plan,output,task,'selected-reference.pt',reference_scope={'role':'meta_train_prefix20_only','preprocessing':mode,'future_queries_read':False})
            else:
                self.complete(plan,output,task,'selected-hyperparameters.pt',selected_config={'preprocessing':mode,'kernel':task['config']['kernel'],'tau':.1,'predictive':'anchored_laplace','anchor_cap':.75,'temperature':1.})
        return plan,output

    def spatial(self,number):
        basis={'key':f'basis-legacy-r4-seed{self.seed}','family':'basis','seed':self.seed,'config':{'rank':4,'variant':'sgd10','head_l2':.001}}
        variants=('sgd10','sgd100','information_protected','tangent_protected') if number==3 else('xfit_select_stack',)
        tasks=[basis]+[{'key':f'core-legacy-legacy-{v}-main-seed{self.seed}','family':'core','seed':self.seed,'variant':v,'basis_key':basis['key'],
                        'config':{'rank':4,'radius':.15,'head_l2':.1}} for v in variants]
        plan,output=self.make_plan(number,tasks);self.complete(plan,output,basis,'selected-basis.pt')
        return plan,output

    def test_import_is_standard_library_only(self):
        script='import sys; import paper3r11.registry; assert "torch" not in sys.modules; assert "numpy" not in sys.modules'
        subprocess.run([sys.executable,'-c',script],check=True,cwd=Path(__file__).resolve().parents[1])

    def test_discovers_r9_profile_and_selected_linear_without_opening_pickle(self):
        self.r9()
        reg=build_registry(self.root)
        self.assertIn(str(self.seed),reg['profile_reference_by_seed'])
        row=reg['baseline_artifacts'][str(self.seed)]['r9_ea_anchored']
        self.assertEqual(reg['baseline_artifacts'][str(self.seed)]['r9_raw_anchored']['r9_config_raw']['kernel'],'rbf')
        self.assertEqual(row['r9_config_ea']['kernel'],'linear')
        self.assertEqual(row['r9_reference_ea']['field'],['reference'])
        self.assertEqual(row['r9_config_ea']['tau'],.1)
        self.assertNotIn('preprocessing',row['r9_config_ea'])
        self.assertNotIn('temperature',row['r9_config_ea'])
        self.assertEqual(row['r9_config_provenance_ea']['role'],'meta_validation')
        self.assertTrue(row['r9_config_provenance_ea']['development_query_labels_used'])
        self.assertFalse(row['r9_config_provenance_ea']['target_query_labels_used'])
        self.assertNotIn('query_labels_used',row['r9_config_provenance_ea'])
        self.assertEqual(validate_registry(reg,seeds=[self.seed])['status'],'READY')
        # Other seeds remain concrete availability diagnostics.
        self.assertTrue(any(r.get('seed')==20261019 for r in reg['missing_artifacts']))

    def test_spatial_follows_declared_legacy_basis_not_holdout_scores(self):
        self.spatial(3);self.spatial(4)
        reg=build_registry(self.root)
        for method in('sgd10','sgd100','r3_protected','r4_xfit_select_stack'):
            row=reg['baseline_artifacts'][str(self.seed)][method]
            self.assertEqual(row['spatial_basis']['field'],['b'])
            self.assertIn('basis-legacy-r4',row['spatial_basis']['path'])
            self.assertEqual(row['basis_provenance']['role'],'meta_train')

    def test_r8_class_reference_legal_role(self):
        task={'key':'class-ref','family':'class_reference','seed':self.seed}
        plan,output=self.make_plan(8,[task])
        self.complete(plan,output,task,'selected-class-reference.pt',reference_scope='meta_train_first20_acquired_prefix_only')
        reg=build_registry(self.root)
        row=reg['baseline_artifacts'][str(self.seed)]['r8_fiber_bounded']
        self.assertEqual(row['class_reference']['field'],['class_reference'])
        self.assertEqual(row['class_reference_provenance']['role'],'meta_train')

    def test_never_recursively_scans_predictions_or_raw_cache(self):
        self.r9()
        raw=self.root/'data/huge_eeg.npy';raw.parent.mkdir(exist_ok=True);raw.write_bytes(b'do-not-read')
        with mock.patch.object(Path,'glob',side_effect=AssertionError('no glob')),mock.patch.object(Path,'rglob',side_effect=AssertionError('no rglob')),mock.patch.object(Path,'walk',create=True,side_effect=AssertionError('no walk')):
            reg=build_registry(self.root)
        self.assertFalse(reg['discovery_policy']['reads_raw_eeg'])
        self.assertFalse(any('huge_eeg' in str(v) for v in reg['discovery_audit']))

    def test_bad_completed_checkpoint_sha_is_missing_not_rewritten(self):
        plan,output=self.r9();path=output/f'feature-reference-ea-seed{self.seed}'/'selected-reference.pt'
        path.write_bytes(b'corrupted')
        reg=build_registry(self.root)
        self.assertNotIn(str(self.seed),reg['profile_reference_by_seed'])
        self.assertEqual(path.read_bytes(),b'corrupted')
        self.assertTrue(any(r['event']=='SELECTED_ARTIFACT_REJECTED' for r in reg['discovery_audit']))

    def test_incomplete_and_query_selected_receipts_rejected(self):
        plan,output=self.r9()
        path=output/f'feature-reference-ea-seed{self.seed}'/'result.json'
        value=json.loads(path.read_text());value['holdout_used_for_selection']=True;self.json(path,value)
        reg=build_registry(self.root)
        self.assertNotIn(str(self.seed),reg['profile_reference_by_seed'])

    def test_explicit_external_manifest_can_be_outside_project(self):
        external=self.root.parent/'r9package/data/pan2023/manifest.json'
        self.json(external,{'dataset_id':'Pan2023','frozen_manifest':'x'})
        task={'key':'external-core','family':'core','seed':self.seed,'partition':'pan2023_published_order','external_manifest':str(external),'external_dataset':'Pan2023'}
        self.make_plan(9,[task])
        row=build_registry(self.root)['external_manifests']['pan2023']
        self.assertEqual(row['path'],str(external))
        self.assertEqual(row['sha256'],hashlib.sha256(external.read_bytes()).hexdigest())

    def test_launch_reference_finds_bounded_backend_plan(self):
        plan,output=self.r9();p=output/'plan.json';nested=output/'runtime/private/native-plan.json';nested.parent.mkdir(parents=True);p.rename(nested)
        self.json(output/'launch.json',{'plan':str(nested)})
        reg=build_registry(self.root)
        self.assertIn(str(self.seed),reg['profile_reference_by_seed'])

    def test_relative_checkpoint_locator_resolves_against_native_output(self):
        plan,output=self.r9();folder=output/f'feature-reference-ea-seed{self.seed}';path=folder/'result.json'
        value=json.loads(path.read_text());value['selected_checkpoint']['path']=f'{folder.name}/selected-reference.pt';self.json(path,value)
        reg=build_registry(self.root)
        self.assertIn(str(self.seed),reg['profile_reference_by_seed'])

    def test_optional_missing_baseline_is_isolated_before_gpu(self):
        self.r9();reg=build_registry(self.root)
        check=validate_registry(reg,stage='formal_exploratory',methods=['r4_xfit_select_stack'],seeds=[self.seed])
        self.assertEqual(check['status'],'READY')
        self.assertEqual(check['missing_artifacts'], [])
        self.assertEqual(check['unavailable_baselines'][0]['method'],'r4_xfit_select_stack')
        self.assertEqual(baseline_availability(reg,'frozen',self.seed)['status'],'AVAILABLE')
        self.assertEqual(baseline_availability(reg,'support_ea',self.seed)['status'],'AVAILABLE')
        self.assertFalse(check['gpu_allocated'])

    def test_writes_only_new_r11_preparation_and_supports_root_api(self):
        self.r9();reg=build_registry(self.root)
        desc=write_registry(reg);path=self.root/desc['path']
        self.assertEqual(json.loads(path.read_text())['schema_version'],'paper3-r11-study-registry-v1')
        self.assertEqual(desc['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
        with self.assertRaises(ValueError):write_registry(reg,self.root/'runs/followup_20261025_r9/plan.json')
        self.assertIn('sha256',write_registry(self.root))

    def test_override_wrong_root_rejected(self):
        with self.assertRaises(ValueError):build_registry(self.root,overrides={'root':'/wrong'})

    def test_no_source_receipts_gives_concrete_paths(self):
        empty=self.root/'empty';empty.mkdir();reg=build_registry(empty)
        row=next(r for r in reg['missing_artifacts'] if r['kind']=='frozen_source_and_reference_receipts')
        self.assertEqual(len(row['expected']),2)

    def test_original_r3_identity_without_separate_recipe_field(self):
        plan,output=self.spatial(3)
        path=output/f'basis-legacy-r4-seed{self.seed}/result.json'
        value=json.loads(path.read_text());value.pop('recipe_sha256')
        from paper3r11.registry import _digest
        value['identity']=_digest({'task':value['task'],'data':value['data_identity'],
            'source':value['source_identity'],'recipe':plan['recipe_sha256'],'basis':None,
            'training':{'updates':1000,'validate_every':50}})
        self.json(path,value)
        reg=build_registry(self.root)
        self.assertIn('sgd10',reg['baseline_artifacts'][str(self.seed)])
        value['identity']='wrong';self.json(path,value)
        self.assertNotIn('sgd10',build_registry(self.root)['baseline_artifacts'][str(self.seed)])

    def test_internal_identity_from_declared_w3_metadata(self):
        reg=build_registry(self.root)
        self.assertEqual(reg['internal_data_identity']['split_sha256'],'b'*64)
        self.assertEqual(reg['internal_data_identity']['signal_contract_sha256'],'c'*64)
        self.assertEqual(reg['participant_split']['path'],'data/participant_split.json')

    def test_override_artifact_sha_mismatch_rejected(self):
        descriptor=describe(self.root/'data/participant_split.json',self.root)
        descriptor['sha256']='0'*64
        with self.assertRaises(ValueError):build_registry(self.root,overrides={'participant_split':descriptor})

    def test_validation_rechecks_changed_dependency_before_gpu(self):
        self.r9();reg=build_registry(self.root)
        path=self.root/reg['profile_reference_by_seed'][str(self.seed)]['checkpoint']['path']
        path.write_bytes(b'changed-after-preparation')
        check=validate_registry(reg,stage='formal_exploratory',methods=['r9_ea_anchored'],seeds=[self.seed])
        self.assertEqual(check['status'],'READY')
        self.assertTrue(any(r['kind']=='artifact_identity_changed' for r in check['unavailable_baselines']))
        self.assertFalse(check['gpu_allocated'])

    def test_repeat_preparation_preserves_immutable_registry_sha(self):
        self.r9();reg=build_registry(self.root)
        first=write_registry(reg)
        repeated=copy.deepcopy(reg);repeated['created_utc']='a different observation time'
        second=write_registry(repeated)
        self.assertEqual(first,second)
        self.assertEqual(repeated['created_utc'],reg['created_utc'])


if __name__=='__main__':unittest.main()
