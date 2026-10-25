from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest

import action_startup as startup

@dataclass(frozen=True)
class Job:
    seed:int=20261018
    method:str='P'


def torch(name='NVIDIA A100-SXM4-40GB', memory=42298834944):
    p=types.SimpleNamespace(name=name,uuid='4e686f4f-128c-7aa2-57eb-54a42ae43d58',total_memory=memory,major=8,minor=0)
    cuda=types.SimpleNamespace(is_available=lambda:True,set_device=lambda i:None,get_device_properties=lambda i:p)
    return types.SimpleNamespace(cuda=cuda,__version__='2.5.1+cu121',version=types.SimpleNamespace(cuda='12.1'))


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.events=[];self.calls=[]
        self.value=(object(),object(),object(),object(),{'inputs':'unchanged'})
        def inputs(root,*,job,source_receipt,selection_receipt=None):
            self.calls.append(('inputs',job,source_receipt));return self.value
        def engineering(root,descriptor,*,job,inputs,runtime_profile=None):
            self.calls.append(('engineering',job,descriptor));return {'proof':descriptor}
        self.inputs=inputs;self.engineering=engineering
        self.modules=(types.SimpleNamespace(verify_meta_inputs=inputs),types.SimpleNamespace(verify_meta_engineering=engineering))
        self.action={'key':'meta-test','kind':'meta_screening'}
    def tearDown(self):self.temp.cleanup()
    def scope(self,**kwargs):
        return startup.active(self.root,self.action,modules=self.modules,torch_module=torch(),emit=lambda event,**fields:self.events.append((event,fields)),traceback_interval=0,**kwargs)
    def test_same_arguments_verify_once_and_reuse_original_objects(self):
        with self.scope():
            a=self.modules[0].verify_meta_inputs(self.root,job=Job(),source_receipt=Path('runs/source.json'))
            b=self.modules[0].verify_meta_inputs(str(self.root),source_receipt='runs/source.json',job=Job(),selection_receipt=None)
            self.assertIs(a,b);self.assertIs(a,self.value)
            proof=self.modules[1].verify_meta_engineering(self.root,{'path':'proof','sha256':'a'},job=Job(),inputs={'x':1})
            self.assertIs(proof,self.modules[1].verify_meta_engineering(self.root,{'sha256':'a','path':'proof'},job=Job(),inputs={'x':1},runtime_profile=None))
        self.assertEqual(len(self.calls),2)
        self.assertEqual(sum(e=='R1_INPUT_VERIFY_CACHE_HIT' for e,_ in self.events),2)
    def test_job_and_descriptor_changes_are_real_checks(self):
        with self.scope():
            self.modules[0].verify_meta_inputs(self.root,job=Job(),source_receipt='source')
            self.modules[0].verify_meta_inputs(self.root,job=Job(seed=20261019),source_receipt='source')
            self.modules[0].verify_meta_inputs(self.root,job=Job(),source_receipt='different-source')
            self.modules[1].verify_meta_engineering(self.root,{'path':'proof','sha256':'a'},job=Job(),inputs={})
            self.modules[1].verify_meta_engineering(self.root,{'path':'proof','sha256':'b'},job=Job(),inputs={})
        self.assertEqual(len(self.calls),5)
    def test_failed_check_is_not_memoized(self):
        attempts=[]
        def failed(root,**kwargs):
            attempts.append(1)
            if len(attempts)==1:raise ValueError('failed validation')
            return object()
        self.modules[0].verify_meta_inputs=failed
        with self.scope():
            with self.assertRaises(ValueError):self.modules[0].verify_meta_inputs(self.root,job=Job())
            v=self.modules[0].verify_meta_inputs(self.root,job=Job())
            self.assertIs(v,self.modules[0].verify_meta_inputs(self.root,job=Job()))
        self.assertEqual(len(attempts),2)
        self.assertIs(self.modules[0].verify_meta_inputs,failed)
    def test_restore_after_action_failure_and_cache_does_not_cross_actions(self):
        with self.assertRaises(KeyboardInterrupt):
            with self.scope():
                self.modules[0].verify_meta_inputs(self.root,job=Job(),source_receipt='source')
                raise KeyboardInterrupt()
        self.assertIs(self.modules[0].verify_meta_inputs,self.inputs)
        self.assertIs(self.modules[1].verify_meta_engineering,self.engineering)
        with self.scope():self.modules[0].verify_meta_inputs(self.root,job=Job(),source_receipt='source')
        self.assertEqual(len(self.calls),2)
    def test_real_sxm40_probe_is_first_and_profiles_are_actual(self):
        with self.scope():self.modules[0].verify_meta_inputs(self.root,job=Job(),source_receipt='source')
        events=[e for e,_ in self.events]
        self.assertLess(events.index('R1_STARTUP_GPU_PROBE_FINISHED'),events.index('R1_INPUT_VERIFY_STARTED'))
        profile=next(v['observed_gpu'] for e,v in self.events if e=='R1_STARTUP_GPU_PROBE_FINISHED')
        self.assertEqual(profile['device_name'],'NVIDIA A100-SXM4-40GB')
        self.assertEqual(profile['device_total_memory_bytes'],42298834944)
    def test_invalid_gpu_rejected_before_any_original_validator(self):
        with self.assertRaisesRegex(ValueError,'observed_gpu'):
            with startup.active(self.root,self.action,modules=self.modules,torch_module=torch(name='NVIDIA V100'),traceback_interval=0,emit=lambda *a,**k:None):
                self.fail('must reject before body')
        self.assertFalse(self.calls)
        self.assertIs(self.modules[0].verify_meta_inputs,self.inputs)
    def test_progress_is_saved_next_to_this_attempt_action_file(self):
        digest=hashlib.sha256(self.action['key'].encode()).hexdigest()
        file=self.root/'runs/training_campaign_parallel_v1/actions'/digest/'attempts'/'claim-token'/'action.json'
        file.parent.mkdir(parents=True);file.write_text('{}')
        with self.scope(action_file=file):
            self.modules[0].verify_meta_inputs(self.root,job=Job(),source_receipt='source')
            value=json.loads(file.with_name('startup-progress.json').read_text())
            self.assertEqual(value['event'],'R1_INPUT_VERIFY_FINISHED')
            self.assertEqual(value['checks_completed'],1)
        final=json.loads(file.with_name('startup-progress.json').read_text())
        self.assertEqual(final['status'],'RETURNED')
        self.assertFalse((self.root/'runs/meta').exists())
    def test_unsupported_argument_is_verified_every_time_not_silently_cached(self):
        def valid(root,*,thing):self.calls.append(thing);return self.value
        self.modules[0].verify_meta_inputs=valid
        thing=object()
        with self.scope():
            self.modules[0].verify_meta_inputs(self.root,thing=thing)
            self.modules[0].verify_meta_inputs(self.root,thing=thing)
        self.assertEqual(len(self.calls),2)
    def test_cpu_selection_does_not_import_torch_or_wrap_native_functions(self):
        action={'key':'selection','kind':'meta_selection'}
        with startup.active(self.root,action,modules=self.modules,torch_module=None,traceback_interval=0,emit=lambda *a,**k:None):
            self.assertIs(self.modules[0].verify_meta_inputs,self.inputs)

if __name__=='__main__':unittest.main()
