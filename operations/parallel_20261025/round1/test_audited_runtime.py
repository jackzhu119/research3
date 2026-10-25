import copy
import json
from pathlib import Path
import tempfile
import types
import unittest

import audited_runtime as migration
from test_hardware_comparison import profile,UUID_A,UUID_B


def pcie_profile():
    value=profile(UUID_B);value['device_name']='NVIDIA A100-PCIE-40GB'
    value['device_total_memory_bytes']=40*1024**3
    return value


class AuthorizedA100MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.native=[]
        self.module=types.SimpleNamespace(
            validate_transfer=lambda *a,**kw:self.native.append((a,kw)) or True,
            _manifest=lambda:{'path':'existing/payload_manifest.json','sha256':'f'*64})
        migration.install(self.root,transfer_module=self.module)

    def tearDown(self):self.temp.cleanup()

    def test_80_to_40_allowed_and_observed_without_mutating_identity(self):
        before=profile();actual=pcie_profile();old=copy.deepcopy(before);new=copy.deepcopy(actual)
        self.assertTrue(self.module.validate_transfer(self.root,before,actual,'native_checkpoint_resume'))
        self.assertFalse(self.native)
        self.assertEqual(before,old);self.assertEqual(actual,new)
        journal=json.loads((self.root/migration.AUDIT).read_text())
        record=next(iter(journal['transfers'].values()))
        self.assertEqual(record['policy']['historical_hardware'],'A100_SXM4_80GB')
        self.assertEqual(record['policy']['actual_hardware'],'A100_PCIE_40GB')
        self.assertEqual(record['actual_runtime']['device_uuid'],UUID_B)
        self.assertFalse(journal['primary_80GB_and_40GB_calibration_timings_may_be_pooled'])

    def test_full_identity_data_code_budget_limits_and_software_must_stay_exact(self):
        before=profile();before.update(code={'native':'sha'},job={'seed':20261018},max_outer_updates=1000)
        actual=pcie_profile();actual.update(code={'native':'sha'},job={'seed':20261018},max_outer_updates=1000)
        self.assertTrue(self.module.validate_transfer(self.root,before,actual,'meta_lineage',record=False))
        for key,new in [('python','3.13'),('torch','2.6'),('numpy','2.3'),('driver_versions',['new']),
                        ('code',{'changed':'sha'}),('job',{'seed':20261019}),('max_outer_updates',2000)]:
            different=copy.deepcopy(actual);different[key]=new
            with self.assertRaises(ValueError):self.module.validate_transfer(self.root,before,different,'changed',record=False)
        self.assertFalse((self.root/migration.AUDIT).exists())

    def test_same_hardware_uuid_only_reuses_existing_hook(self):
        before=profile();actual=profile(UUID_B)
        self.assertTrue(self.module.validate_transfer(self.root,before,actual,'equivalent'))
        self.assertEqual(len(self.native),1)

    def test_driver_version_inventory_cardinality_is_not_a_version_change(self):
        before=profile();before['driver_versions']=['535.104.05']*8
        actual=pcie_profile();actual['driver_versions']=['535.104.05']*4
        self.assertTrue(self.module.validate_transfer(self.root,before,actual,'different_node_card_count'))
        actual['driver_versions']=['535.104.05']*3+['550.1']
        with self.assertRaises(ValueError):self.module.validate_transfer(self.root,before,actual,'different_driver')

    def test_unsupported_architecture_model_or_memory_is_rejected(self):
        for key,new in [('device_name','NVIDIA H100'),('device_compute_capability',[9,0]),
                        ('device_total_memory_bytes',20*1024**3)]:
            actual=pcie_profile();actual[key]=new
            with self.assertRaises(ValueError):migration.validate_typed_transfer(profile(),actual)

    def test_historical_development_measurement_is_retained_and_needs_remeasurement(self):
        out=self.root/'runs/training_campaign_v1/profiles/B1/receipt.json';out.parent.mkdir(parents=True)
        value={'status':'DEVELOPMENT_CALIBRATION_PROFILE_COMPLETE','runtime_profile':profile()}
        out.write_text(json.dumps(value));before=out.read_bytes()
        report=migration.inspect_primary_measurements(self.root)
        self.assertFalse(report['measurement_hardware_matches'])
        row=report['incompatible_measurements'][0]
        self.assertTrue(row['target_calibration_remeasurement_required'])
        self.assertFalse(row['source_or_meta_retraining_required'])
        self.assertEqual(out.read_bytes(),before)

    def test_completed_40gb_measurements_match_the_new_target(self):
        out=self.root/'runs/training_campaign_v1/development/job/receipt.json';out.parent.mkdir(parents=True)
        out.write_text(json.dumps({'status':'DEVELOPMENT_EVALUATION_COMPLETE','inputs':{'runtime_profile':pcie_profile()}}))
        self.assertTrue(migration.inspect_primary_measurements(self.root)['measurement_hardware_matches'])


if __name__=='__main__':unittest.main()
