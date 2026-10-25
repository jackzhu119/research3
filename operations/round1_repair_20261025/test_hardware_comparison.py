import ast
import copy
from dataclasses import asdict,dataclass
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import types
import unittest

import hardware_comparison as hw
import kernel_bridge as bridge

NATIVE=Path('/workspace/paper3_hpc')
UUID_A='11111111-1111-1111-1111-111111111111'
UUID_B='22222222-2222-2222-2222-222222222222'


def profile(uuid=UUID_A):
    return {'python':'3.12.3','torch':'2.5.1+cu121','numpy':'2.2.6','cuda':'12.1',
            'device_name':'NVIDIA A100-SXM4-80GB','device_uuid':uuid,
            'device_total_memory_bytes':80*1024**3,'device_compute_capability':[8,0],
            'driver_versions':['535.104.05'],'autocast':False,'tf32':False}


class HardwareComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'p3work').mkdir()
        for name in ('development_evaluation.py','internal_validation.py','method_registry.py'):
            shutil.copyfile(NATIVE/'p3work'/name,self.root/'p3work'/name)
        self.calls=[]
        self.dev=types.ModuleType('fake_native_development')
        d=self.dev.__dict__
        registry=types.SimpleNamespace(registered_grid=bridge._grid_function(self.root))
        d.update(Path=Path,dataclass=dataclass,asdict=asdict,registry=registry,METHODS=bridge.METHODS,
            CONTROLS=tuple('B%d'%i for i in range(1,12)),SEEDS=bridge.SEEDS,
            SCHEMA='paper3-development-evaluation-v1',DevelopmentGateError=ValueError,
            CONFIGURATION_TIE_RULE='participant_BA_desc/balanced_BCE_asc/selected_meta_checkpoint_asc/registered_grid_order',
            digest=hw._digest,re=__import__('re'))
        def local(root,path,exists=True):return Path(root)/path
        def reference(root,item,suffix=None):
            p=Path(root)/item['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'];return p
        def put(path,value):
            path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():self.assertEqual(json.loads(path.read_text()),value)
            else:path.write_text(json.dumps(value,sort_keys=True))
        def verify(root,item):
            self.calls.append(item['path']);value=json.loads(reference(root,item).read_text())
            if value.get('native_proof_valid') is not True:raise ValueError('Native verification rejected')
            return value
        d.update(_local=local,_reference=reference,_json=lambda p:json.loads(Path(p).read_text()),_put=put,
                 verify_development_receipt=verify)
        self.original_verify=verify
        wanted={'DevelopmentJob','freeze_screening_configuration','verify_configuration_receipt','select_strongest_control'}
        nodes=[n for n in ast.parse((NATIVE/'p3work/development_evaluation.py').read_bytes()).body
               if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in wanted]
        sel=next(n for n in ast.parse((NATIVE/'p3work/selection.py').read_bytes()).body
                 if isinstance(n,ast.FunctionDef) and n.name=='choose_strongest_control')
        d.update(defaultdict=defaultdict,math=math)
        exec(compile(ast.Module(body=[sel]+nodes,type_ignores=[]),'native-selection','exec'),d)
        self.internal=types.ModuleType('fake_native_internal')
        self.internal.__dict__.update(Path=Path,OUTPUT='runs/internal_validation_once')

    def tearDown(self):self.temp.cleanup()

    def save(self,path,value):
        p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,sort_keys=True))
        return {'path':path,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

    def receipt(self,method,index,seed,uuid=UUID_A,configuration_receipt=None):
        configuration=self.dev.registry.registered_grid(method)[index]
        runtime=profile(uuid)
        value={'native_proof_valid':True,'status':'DEVELOPMENT_EVALUATION_COMPLETE',
            'synthetic':False,'method':method,'seed':seed,'role':'meta_validation','measured':True,
            'scientific_fit_completed':True,'configuration':configuration,
            'job':{'method':method,'configuration_index':index,'seed':seed,
                   'phase':'screening' if seed==bridge.SEEDS[0] else 'confirmation'},
            'catalog':{'catalog':'frozen'},'inputs':{'runtime_profile':runtime,
                'device_identity_sha256':hw._digest(runtime),'code':{'same':'code'},
                'dependencies':{'source_f0':{'path':'source-%d'%seed,'sha256':'a'*64}},
                'configuration_receipt':configuration_receipt,'selected_meta_outer_update':1000},
            'participant_mean_ba':.6+index*.001,'participant_mean_bce':.7-index*.001,
            'calibration_median_seconds':1.+int(method[1:])*.01,'timing_protocol':'fixed-protocol'}
        return self.save('receipts/%s-%d-%d.json'%(method,index,seed),value)

    def install(self):hw.install(self.root,development_module=self.dev,internal_module=self.internal)

    def test_native_freeze_and_verify_roundtrip_accepts_uuid_only_move(self):
        refs=[self.receipt('B1',i,bridge.SEEDS[0],UUID_A if i<4 else UUID_B) for i in range(9)]
        before={p:p.read_bytes() for p in self.root.rglob('*.json')}
        with self.assertRaises(ValueError):
            self.dev.freeze_screening_configuration(self.root,method='B1',screening_receipts=refs)
        self.install()
        out='runs/training_campaign_v1/selections/B1.json'
        result=self.dev.freeze_screening_configuration(self.root,method='B1',screening_receipts=refs,output=out,execute=True)
        descriptor={'path':out,'sha256':hashlib.sha256((self.root/out).read_bytes()).hexdigest()}
        self.assertEqual(self.dev.verify_configuration_receipt(self.root,descriptor),result)
        self.assertEqual(result['configuration_index'],8)
        for p,data in before.items():self.assertEqual(p.read_bytes(),data)
        audit=json.loads((self.root/hw.AUDIT).read_text())
        self.assertEqual({r['runtime_profile']['device_uuid'] for r in audit['observations'].values()},{UUID_A,UUID_B})

    def test_native_verification_is_never_skipped(self):
        ref=self.receipt('B1',0,bridge.SEEDS[0])
        p=self.root/ref['path'];value=json.loads(p.read_text());value['native_proof_valid']=False;ref=self.save(ref['path'],value)
        self.install()
        with self.assertRaisesRegex(ValueError,'Native verification rejected'):
            self.dev.verify_development_receipt(self.root,ref)
        self.assertEqual(self.calls,[ref['path']])
        self.assertFalse((self.root/hw.AUDIT).exists())

    def test_model_torch_numpy_driver_mismatch_is_not_normalized(self):
        self.install();ref=self.receipt('B1',0,bridge.SEEDS[0])
        first=self.dev.verify_development_receipt(self.root,ref)
        self.assertEqual(first['inputs']['runtime_profile']['device_uuid'],UUID_A)
        for field,changed in [('device_name','NVIDIA A100-PCIE-40GB'),('torch','9.9'),('numpy','9.9'),
                              ('driver_versions',['different']),('device_total_memory_bytes',40*1024**3)]:
            value=json.loads((self.root/ref['path']).read_text());value['inputs']['runtime_profile'][field]=changed
            value['inputs']['device_identity_sha256']=hw._digest(value['inputs']['runtime_profile'])
            bad=self.save('bad-'+field+'.json',value)
            with self.assertRaises(ValueError):self.dev.verify_development_receipt(self.root,bad)

    def test_original_runtime_copy_is_unmodified(self):
        ref=self.receipt('B1',0,bridge.SEEDS[0]);raw=self.original_verify(self.root,ref);before=copy.deepcopy(raw)
        normalized=hw.normalize_verified(self.root,raw,ref)
        self.assertEqual(raw,before)
        self.assertNotEqual(normalized['inputs']['device_identity_sha256'],raw['inputs']['device_identity_sha256'])
        self.assertEqual(normalized['inputs']['observed_device_identity_sha256'],raw['inputs']['device_identity_sha256'])
        self.assertEqual(normalized['inputs']['runtime_profile'],raw['inputs']['runtime_profile'])

    def test_driver_node_inventory_count_does_not_change_hardware_class(self):
        before=profile();before['driver_versions']=['535.104.05']*8
        actual=profile(UUID_B);actual['driver_versions']=['535.104.05']*4
        self.assertTrue(hw.equivalent_runtime(self.root,before,actual,'different_node_inventory'))
        self.assertEqual(before['driver_versions'],['535.104.05']*8)
        self.assertEqual(actual['driver_versions'],['535.104.05']*4)
        actual['driver_versions']=['550.1']*4
        with self.assertRaises(ValueError):hw.equivalent_runtime(self.root,before,actual,'driver_changed')

    def test_strongest_control_legacy_bytes_and_rebuild_are_preserved(self):
        configs={};confirmations=[]
        for method in self.dev.CONTROLS:
            refs=[self.receipt(method,i,bridge.SEEDS[0]) for i in range(9)]
            output='runs/training_campaign_v1/selections/'+method+'.json'
            self.dev.freeze_screening_configuration(self.root,method=method,screening_receipts=refs,output=output,execute=True)
            config={'path':output,'sha256':hashlib.sha256((self.root/output).read_bytes()).hexdigest()}
            configs[method]=config;confirmations.append(refs[8])
            confirmations.extend(self.receipt(method,8,seed,configuration_receipt=config) for seed in bridge.SEEDS[1:])
        out='runs/training_campaign_v1/strongest_control.json'
        prior=self.dev.select_strongest_control(self.root,configuration_receipts=configs,confirmation_receipts=confirmations,output=out,execute=True)
        before=(self.root/out).read_bytes()
        self.install()
        rebuilt=self.dev.select_strongest_control(self.root,configuration_receipts=configs,confirmation_receipts=confirmations)
        self.assertEqual(rebuilt,prior)
        again=self.dev.select_strongest_control(self.root,configuration_receipts=configs,confirmation_receipts=confirmations,output=out,execute=True)
        self.assertEqual(again,prior);self.assertEqual((self.root/out).read_bytes(),before)

    def test_internal_claim_lineage_retains_original_bytes(self):
        old={'freeze':{'x':1},'budget':{'same':True},'output':'runs/internal_validation_once','runtime':profile(UUID_A)}
        claim=self.save('runs/internal_validation_once_claim.json',{'identity':old})
        raw=(self.root/claim['path']).read_bytes();new=copy.deepcopy(old);new['runtime']=profile(UUID_B)
        result=hw._internal_identity(self.root,self.root/'runs/internal_validation_once',new)
        self.assertEqual(result,old);self.assertEqual((self.root/claim['path']).read_bytes(),raw)
        changed=copy.deepcopy(new);changed['budget']={'different':True}
        with self.assertRaises(ValueError):hw._internal_identity(self.root,self.root/'runs/internal_validation_once',changed)

    def test_new_mixed_uuid_control_freeze_and_rebuild_close(self):
        configs={};confirmations=[]
        for method in self.dev.CONTROLS:
            refs=[self.receipt(method,i,bridge.SEEDS[0]) for i in range(9)]
            output='runs/training_campaign_v1/selections/'+method+'.json'
            self.dev.freeze_screening_configuration(self.root,method=method,screening_receipts=refs,output=output,execute=True)
            config={'path':output,'sha256':hashlib.sha256((self.root/output).read_bytes()).hexdigest()}
            configs[method]=config;confirmations.append(refs[8])
            for seed in bridge.SEEDS[1:]:
                confirmations.append(self.receipt(method,8,seed,UUID_B if seed==bridge.SEEDS[2] else UUID_A,config))
        with self.assertRaises(ValueError):
            self.dev.select_strongest_control(self.root,configuration_receipts=configs,confirmation_receipts=confirmations)
        self.install();out='runs/training_campaign_v1/strongest_control.json'
        result=self.dev.select_strongest_control(self.root,configuration_receipts=configs,
            confirmation_receipts=confirmations,output=out,execute=True)
        self.assertEqual(len({r['device_identity_sha256'] for r in result['evidence_rows']}),1)
        rebuilt=self.dev.select_strongest_control(self.root,configuration_receipts=configs,confirmation_receipts=confirmations)
        self.assertEqual(rebuilt,result)

    def test_internal_runtime_comparisons_are_exact_administrative_edits(self):
        self.install()
        import dis
        for name in ('execute_internal','_load_fit'):
            self.assertIn('_paper3_parallel_equivalent_runtime',getattr(self.internal,name).__code__.co_names)
        self.assertIn('_paper3_parallel_internal_identity',self.internal.execute_internal.__code__.co_names)
        self.assertEqual(hashlib.sha256((self.root/'p3work/internal_validation.py').read_bytes()).hexdigest(),hw.PINNED['p3work/internal_validation.py'])



class HardwareAliasCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        hw._OBSERVED_CACHE.clear()
    def tearDown(self):self.temp.cleanup()
    def test_same_full_device_name_alias_retains_actual_observations(self):
        before=profile(UUID_A);actual=profile(UUID_B)
        actual['device_name']='nvidia A100 SXM4 80GB'
        raw_before=copy.deepcopy(before);raw_actual=copy.deepcopy(actual)
        self.assertTrue(hw.equivalent_runtime(self.root,before,actual,'name_alias'))
        self.assertEqual(before,raw_before);self.assertEqual(actual,raw_actual)
        journal=json.loads((self.root/hw.AUDIT).read_text())
        self.assertEqual({r['runtime_profile']['device_name'] for r in journal['observations'].values()},
            {'NVIDIA A100-SXM4-80GB','nvidia A100 SXM4 80GB'})
    def test_actual_school_sxm4_40gb_alias_preserves_native_hash(self):
        actual=profile(UUID_A)
        actual.update(device_name='NVIDIA A100-SXM4-40GB',device_total_memory_bytes=42298834944,
                      driver_versions=['535.104.12']*4)
        original=copy.deepcopy(actual);raw_hash=hw._digest(actual)
        value={'inputs':{'runtime_profile':actual,'device_identity_sha256':raw_hash}}
        normalized=hw.normalize_verified(self.root,value,{'path':'native/receipt.json'})
        self.assertEqual(normalized['inputs']['runtime_profile'],original)
        self.assertEqual(normalized['inputs']['observed_device_identity_sha256'],raw_hash)
        alias=copy.deepcopy(actual);alias.update(device_name='A100 SXM4 40GB',device_uuid=UUID_B,
                                               driver_versions=['535.104.12']*8)
        self.assertTrue(hw.equivalent_runtime(self.root,actual,alias,'school_sxm40_alias'))
        self.assertEqual(actual,original)
    def test_old_raw_name_journal_retains_original_comparison_identity(self):
        before=profile(UUID_A);raw_class=hw._raw_profile(before)
        original_id=hw._digest({'schema_version':hw.SCHEMA,'runtime_profile_without_device_uuid':raw_class})
        old_record={'runtime_profile':copy.deepcopy(before),'native_runtime_identity_sha256':hw._digest(before),'reference':{'old':True}}
        value={'schema_version':hw.SCHEMA,'hardware_class_profile':raw_class,
               'hardware_class_identity_sha256':original_id,'observations':{'old':old_record}}
        path=self.root/hw.AUDIT;path.parent.mkdir(parents=True);path.write_text(json.dumps(value))
        actual=profile(UUID_B);actual['device_name']='A100 SXM4 80GB'
        first=hw.observe(self.root,actual,{'new':True})
        second=hw.observe(self.root,actual,{'new':True})
        self.assertEqual(first,original_id);self.assertEqual(second,original_id)
        after=json.loads(path.read_text())
        self.assertEqual(after['hardware_class_profile'],raw_class)
        self.assertEqual(after['hardware_class_identity_sha256'],original_id)
        self.assertEqual(after['observations']['old'],old_record)
    def test_existing_wrong_hardware_class_still_cannot_pool(self):
        hw.observe(self.root,profile(UUID_A),{'first':True})
        for name,memory in [('NVIDIA A100-PCIE-80GB',80),('NVIDIA A100-SXM4-40GB',40)]:
            candidate=profile(UUID_B);candidate.update(device_name=name,device_total_memory_bytes=memory*1024**3)
            with self.assertRaises(ValueError):hw.observe(self.root,candidate,{'new':True})
    def test_memory_and_software_are_not_normalized_as_names(self):
        before=profile(UUID_A);actual=profile(UUID_B);actual['device_name']='A100 SXM4 80GB'
        for key,value in [('device_total_memory_bytes',80*1024**3-1024),('torch','2.6'),('numpy','9.9')]:
            changed=copy.deepcopy(actual);changed[key]=value
            with self.assertRaises(ValueError):hw.equivalent_runtime(self.root,before,changed,'different')

if __name__=='__main__':unittest.main()
