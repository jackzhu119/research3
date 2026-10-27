import ast
from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import kernel_bridge as bridge

NATIVE = Path('/workspace/paper3_hpc')


class KernelBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/'p3work').mkdir()
        shutil.copyfile(NATIVE/'p3work/method_registry.py',self.root/'p3work/method_registry.py')
        for path,status in [
            ('runs/controlled_source_pilot_seed20261018/engineering_receipt.json','ENGINEERING_PILOT_COMPLETE'),
            ('runs/source_training_v2/source_profile_development.json','SOURCE_PROFILE_NEURAL_COMPLETE_CLASSICAL_PENDING'),
            ('runs/source_classical_v2/receipt.json','SOURCE_CLASSICAL_CORE_COMPLETE'),
        ] + [(bridge.OUTPUT+'/references/seed-%s/receipt.json'%s,'SOURCE_REFERENCE_COMPLETE') for s in bridge.SEEDS]:
            self.write(path,{'status':status})

    def tearDown(self):
        self.temp.cleanup()

    def write(self,path,value):
        p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))

    def complete(self,action):
        value={'status':bridge.STATUS[action['kind']]}
        if action['kind']=='extension_initialize':
            value={'schema_version':'native','arm':action['arm']}
        elif action['kind']=='meta_selection':
            config=bridge._grid_function(self.root)(action['method'])[0]
            value.update(dimension=config.get('dimension',config.get('rank')),radius=config['radius'])
        elif action['kind']=='development_selection':
            value.update(configuration_index=0,configuration=bridge._grid_function(self.root)(action['method'])[0])
        self.write(action['expected_receipt'],value)

    def test_metadata_dag_phase_barriers_and_all_original_extensions(self):
        p=bridge.plan_ready(self.root)
        self.assertEqual(p['phase'],'META_SCREENING')
        self.assertEqual(len(p['ready']),5)
        self.assertEqual(set(p['total_counts']),{'meta_engineering','meta_screening','meta_selection'})
        self.assertEqual(p['total_counts']['meta_screening'],45)
        for a in p['ready']:self.complete(a)
        p=bridge.plan_ready(self.root);self.assertEqual(len(p['ready']),45)
        for a in p['ready']:self.complete(a)
        p=bridge.plan_ready(self.root);self.assertEqual(len(p['ready']),5)
        for a in p['ready']:self.complete(a)
        p=bridge.plan_ready(self.root);self.assertEqual(p['phase'],'DEVELOPMENT_SCREENING')
        self.assertEqual(p['total_counts']['development'],118)
        self.assertEqual(len(p['ready']),14)
        self.assertFalse(bridge.plan_ready(self.root,slot=1)['ready'])
        for a in p['ready']:self.complete(a)
        p=bridge.plan_ready(self.root);self.assertEqual(len(p['ready']),118)
        for a in p['ready']:self.complete(a)
        p=bridge.plan_ready(self.root);self.assertEqual(len(p['ready']),14)
        for a in p['ready']:self.complete(a)
        p=bridge.plan_ready(self.root);self.assertEqual(p['phase'],'META_CONFIRMATION')
        self.assertEqual(len(p['ready']),10)
        self.assertEqual(len(bridge.plan_ready(self.root,slot=2)['ready']),10)
        self.complete(p['ready'][0])
        p=bridge.plan_ready(self.root);self.assertEqual(len(p['ready']),9)
        self.assertFalse(any(a['kind']=='development' for a in p['ready']))
        for a in p['ready']:self.complete(a)
        p=bridge.plan_ready(self.root);self.assertEqual(p['phase'],'DEVELOPMENT_CONFIRMATION')
        self.assertEqual(len(p['ready']),28)
        self.assertFalse(bridge.plan_ready(self.root,slot=2)['ready'])
        for a in p['ready']:self.complete(a)
        for kind in ('strongest_control','freeze','internal'):
            p=bridge.plan_ready(self.root);self.assertEqual([a['kind'] for a in p['ready']],[kind])
            self.complete(p['ready'][0])
        p=bridge.plan_ready(self.root);self.assertTrue(p['mainline_complete'])
        self.assertEqual(len(p['ready']),2)
        self.assertEqual([a['affinity_slot'] for a in p['ready']],[1,2])
        for a in p['ready']:self.complete(a)
        p=bridge.plan_ready(self.root);self.assertEqual(len(p['ready']),2)
        self.assertEqual(set(a['kind'] for a in p['ready']),{'extension_development'})
        for a in p['ready']:self.complete(a)
        self.assertTrue(bridge.plan_ready(self.root)['all_complete'])

    def test_native_job_identities_match_all_registered_grid_jobs(self):
        meta_source=ast.parse((NATIVE/'p3work/meta_training.py').read_bytes())
        cls=next(n for n in meta_source.body if isinstance(n,ast.ClassDef) and n.name=='MetaJob')
        ns={'dataclass':dataclass,'digest':bridge._digest,'asdict':asdict}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'native-meta','exec'),ns)
        dev_source=ast.parse((NATIVE/'p3work/development_evaluation.py').read_bytes())
        devcls=next(n for n in dev_source.body if isinstance(n,ast.ClassDef) and n.name=='DevelopmentJob')
        registry=type('Registry',(),{'registered_grid':staticmethod(bridge._grid_function(self.root))})
        ds={'dataclass':dataclass,'digest':bridge._digest,'asdict':asdict,'METHODS':bridge.METHODS,
            'SEEDS':bridge.SEEDS,'registry':registry,'SCHEMA':'paper3-development-evaluation-v1','DevelopmentGateError':ValueError}
        exec(compile(ast.Module(body=[devcls],type_ignores=[]),'native-development','exec'),ds)
        grid=bridge._grid_function(self.root)
        for method in bridge.LEARNED:
            for config in grid(method):
                for seed in bridge.SEEDS:
                    job=bridge._meta_job(method,config,seed)
                    self.assertEqual(ns['MetaJob'](**job).identity,bridge._digest(job))
        for method in bridge.METHODS:
            for i,c in enumerate(grid(method)):
                for seed in bridge.SEEDS:
                    job,identity=bridge._development_job(method,i,c,seed)
                    self.assertEqual(ds['DevelopmentJob'](**job).identity,identity)

    def test_complete_receipt_never_activates_kernel_or_changes_source(self):
        action=bridge.plan_ready(self.root)['ready'][0];self.complete(action)
        before={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with mock.patch.object(bridge,'activate_runtime',side_effect=AssertionError('No kernel import permitted')):
            result=bridge.perform_action(self.root,action,self.root/'unused-manifest.json')
        self.assertEqual(result['status'],'REUSED_COMPLETE')
        self.assertEqual(before,{p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_planning_cache_is_one_pass_and_observes_new_receipts(self):
        a=bridge.plan_ready(self.root)['ready'][0]
        first=bridge.plan_ready(self.root)
        self.complete(a)
        second=bridge.plan_ready(self.root)
        self.assertNotEqual(first['phase_completed'],second['phase_completed'])
        self.assertIsNone(bridge._READ_CACHE.get())

    def test_noncomplete_old_receipt_is_not_counted_as_success(self):
        a=bridge.plan_ready(self.root)['ready'][0]
        self.write(a['expected_receipt'],{'status':'INTERRUPTED_NOT_COMPLETE'})
        self.assertFalse(bridge.action_complete(self.root,a))
        self.assertIn(a['key'],[x['key'] for x in bridge.plan_ready(self.root)['ready']])

    def test_missing_completed_prerequisites_are_pending_without_retraining(self):
        (self.root/'runs/source_classical_v2/receipt.json').unlink()
        result=bridge.plan_ready(self.root)
        self.assertEqual(result['phase'],'EXISTING_PREREQUISITES_PENDING')
        self.assertFalse(result['ready'])

    def test_on_ready_occurs_after_activation_before_native_perform(self):
        action=bridge.plan_ready(self.root)['ready'][0]
        events=[];owner=self
        class Production:
            def __init__(self,root,maximum):
                self.root=root
                owner.assertGreater(maximum,529200)
            def perform(self,native):
                owner.assertEqual(events,['activated','ready'])
                owner.assertNotIn('affinity_slot',native)
                owner.assertNotIn('serial',native)
                events.append('native')
                owner.complete(action)
                return {'path':action['expected_receipt'],'sha256':'fixture'}
        campaign=type('Native',(),{'Production':Production,
            'validate_registered_action':staticmethod(lambda a:a)})
        def activate(*a,**kw):events.append('activated');return campaign
        with mock.patch.object(bridge,'activate_runtime',side_effect=activate):
            result=bridge.perform_action(self.root,action,self.root/'payload.json',on_ready=lambda:events.append('ready'))
        self.assertEqual(events,['activated','ready','native'])
        self.assertEqual(result['status'],'COMPLETE')


if __name__=='__main__':unittest.main()
