"""Pure software/structural fixtures for the versioned W3 verifier.

No real cache, episode, raw EEG, network or fitting is opened by these tests.
The production software-history loader has its own strict contract tests; the
small mocked history below isolates the extra 114+3 verifier checks.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from p3work import episodes
from tools import audit_completed_stieger_w3 as original
from tools import audit_completed_stieger_w3_v2 as audit
from tools import build_data_plan_deploy_manifest as deploy
from tools import build_data_plan_deploy_manifest_v2 as deployment_v2

REPO=Path(__file__).resolve().parents[1]
SCOPE='audit/data_plan_20261020/episode_descriptor_compatibility_fix'
EPISODE_SNAPSHOT=SCOPE+'/original_episodes_348604.py.txt'
PREDICTION_SNAPSHOT=SCOPE+'/original_prediction_store_e1af63.py.txt'
SOURCE_TRAINING_SNAPSHOT='hpc/fixtures/original_source_training_54b972.py.txt'
SOURCE_TRAINING_HISTORY={'path':'p3work/source_training.py',
    'sha256':'54b972457ea1bbc08c15aeff9440a418047b7752b725dac3c27d562bead2a849','bytes':80473}
PILOT_SNAPSHOT='hpc/fixtures/original_run_controlled_pilot_856393.py.txt'
PILOT_HISTORY={'path':'tools/run_controlled_pilot.py',
    'sha256':'856393030d10cdbbee54138fefadce6ee8520243f9e54d00feb7d5328088172c','bytes':50853}


def historical_source_training(manifest, snapshot=None, *, expected=SOURCE_TRAINING_HISTORY):
    """Bind the original historical-verifier fixture to its genuine snapshot."""
    historical=[row for row in manifest['files'] if row['path']==expected['path']]
    if historical != [expected]:
        raise AssertionError('exact_original_source_training_inventory_required')
    body=(snapshot or REPO/SOURCE_TRAINING_SNAPSHOT).read_bytes()
    actual={'path':expected['path'],'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body)}
    if actual != expected:
        raise AssertionError('historical_source_training_snapshot_SHA_or_bytes_mismatch')
    return body


def function_ast(path,name):
    tree=ast.parse(path.read_text())
    return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)


class HistoricalGateStructureTests(unittest.TestCase):
    def test_frozen_parent_and_canonical_consumer_are_retained(self):
        body=(REPO/audit.PREVIOUS_VERIFIER).read_bytes()
        self.assertEqual(hashlib.sha256(body).hexdigest(),audit.PREVIOUS_VERIFIER_SHA)
        self.assertIs(audit.load_episode_catalog,episodes.load_episode_catalog)
        self.assertNotEqual(audit.OUTPUT,original.OUTPUT)

    def test_transfer_accounting_and_clock_helpers_are_exact_parent_AST(self):
        for name in ('pinned','bound','finite_nonnegative','session_accounting'):
            self.assertEqual(ast.dump(function_ast(REPO/audit.PREVIOUS_VERIFIER,name)),
                ast.dump(function_ast(REPO/'tools/audit_completed_stieger_w3_v2.py',name)))

    def test_cpu_gate_changes_only_authenticated_history_registration(self):
        old=function_ast(REPO/audit.PREVIOUS_VERIFIER,'cpu_completion')
        new=function_ast(REPO/'tools/audit_completed_stieger_w3_v2.py','cpu_completion')
        registrations=[]
        class Normalize(ast.NodeTransformer):
            def visit_Expr(self,node):
                call=node.value
                if (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                        and call.func.attr=='register_episode_history'):
                    registrations.append(ast.dump(call))
                    if (call.args or len(call.keywords)!=1 or call.keywords[0].arg!='defer_walk'
                            or not isinstance(call.keywords[0].value,ast.Constant)
                            or call.keywords[0].value.value is not True):
                        raise AssertionError('exact_deferred_history_registration_required')
                    return None
                return self.generic_visit(node)
            def visit_Attribute(self,node):
                if isinstance(node.value,ast.Name) and node.value.id=='deploy_v2' and node.attr=='Closure':
                    node.value.id='deploy'
                return self.generic_visit(node)
        normalized=Normalize().visit(new)
        self.assertEqual(len(registrations),1)
        self.assertEqual(ast.dump(old),ast.dump(normalized))

    def test_all_real_data_CPU_duplicate_and_episode_gates_are_exact_parent_AST(self):
        def gates(path):
            body=function_ast(path,'main').body
            first=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='software' for t in n.targets))
            last=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='report' for t in n.targets))
            return ast.dump(ast.Module(body=body[first:last],type_ignores=[]))
        self.assertEqual(gates(REPO/audit.PREVIOUS_VERIFIER),
            gates(REPO/'tools/audit_completed_stieger_w3_v2.py'))


class VersionedSoftwareIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        manifest=json.loads((REPO/'audit/software_code_manifest.json').read_bytes())
        paths={r['path'] for r in manifest['files']}
        paths.update({'audit/software_code_manifest.json',audit.HISTORICAL_VERIFIER,audit.PREVIOUS_VERIFIER,
            audit.PRECISION_AMENDMENT,audit.FROZEN_CLOSURE,EPISODE_SNAPSHOT,PREDICTION_SNAPSHOT,
            'audit/data_plan_20261019/stieger/cache_plan.json'})
        paths.update(deploy.PRECISION_AUDIT+'/'+p for p in
            ('original_classical.py','original_software_code_manifest.json','native_regression.json'))
        for relative in paths:
            target=self.root/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if relative=='p3work/source_training.py':
                target.write_bytes(historical_source_training(manifest))
                continue
            if relative=='tools/run_controlled_pilot.py':
                target.write_bytes(historical_source_training(manifest,REPO/PILOT_SNAPSHOT,expected=PILOT_HISTORY))
                continue
            shutil.copyfile(REPO/relative,target)
        contract=self.write('contracts/episode_descriptor_compatibility_20261020.json',
            {'software_fixture_only':True,'scientific_fits':0})
        acceptance=self.write('audit/fixture_compatibility_acceptance.json',
            {'software_fixture_only':True,'real_full_suite_claimed':False})
        self.history={'historical_source_path':audit.EPISODE_SOURCE,
            'historical_source_sha256':audit.EPISODE_OLD_SHA,'historical_source_bytes':43688,
            'archive_snapshot':deploy.descriptor(self.root,EPISODE_SNAPSHOT),
            'current_implementation':deploy.descriptor(self.root,audit.EPISODE_SOURCE),
            'software_fix_contract':contract,'actual_new_software_acceptance':acceptance,
            'original_117_manifest':deploy.descriptor(self.root,'audit/software_code_manifest.json'),
            'prediction_store_history':{'historical_source_path':audit.PREDICTION_SOURCE,
                'historical_source_sha256':audit.PREDICTION_OLD_SHA,'historical_source_bytes':32085,
                'archive_snapshot':deploy.descriptor(self.root,PREDICTION_SNAPSHOT),
                'current_implementation':deploy.descriptor(self.root,audit.PREDICTION_SOURCE)}}

    def write(self,path,value):
        p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(value if isinstance(value,bytes) else json.dumps(value).encode())
        return deploy.descriptor(self.root,path)

    def inspect(self):
        with patch.object(deployment_v2.Closure,'register_episode_history',
                return_value=copy.deepcopy(self.history),create=True) as registration:
            result=audit.software_integrity(self.root)
            registration.assert_called_once_with(defer_walk=True)
            return result

    def test_exact_114_and_three_declared_current_fixes_pass(self):
        report=self.inspect()
        self.assertEqual(report['historical_unchanged_files'],114)
        self.assertEqual(report['approved_modified_files'],3)
        self.assertEqual(report['descriptor_consumer_compatibility_history'],self.history)
        self.assertNotIn('historical_116_code_SHA_unchanged',report)

    def test_historical_source_training_snapshot_tampering_is_rejected_before_copy(self):
        manifest=json.loads((REPO/'audit/software_code_manifest.json').read_bytes())
        snapshot=self.root/'tampered-original-source-training.txt'
        snapshot.write_bytes((REPO/SOURCE_TRAINING_SNAPSHOT).read_bytes()+b'\n# changed\n')
        with self.assertRaisesRegex(AssertionError,'snapshot_SHA_or_bytes_mismatch'):
            historical_source_training(manifest,snapshot)

    def test_historical_source_training_inventory_remains_exact(self):
        manifest=json.loads((REPO/'audit/software_code_manifest.json').read_bytes())
        row=next(row for row in manifest['files'] if row['path']=='p3work/source_training.py')
        row['bytes']+=1
        with self.assertRaisesRegex(AssertionError,'exact_original_source_training_inventory'):
            historical_source_training(manifest)

    def test_historical_pilot_archive_tampering_is_rejected_before_copy(self):
        manifest=json.loads((REPO/'audit/software_code_manifest.json').read_bytes())
        snapshot=self.root/'tampered-original-pilot.txt'
        snapshot.write_bytes((REPO/PILOT_SNAPSHOT).read_bytes()+b'\n# changed\n')
        with self.assertRaisesRegex(AssertionError,'snapshot_SHA_or_bytes_mismatch'):
            historical_source_training(manifest,snapshot,expected=PILOT_HISTORY)

    def test_fourth_original_file_change_is_rejected(self):
        self.write('p3work/__init__.py',b'unapproved fourth change\n')
        with self.assertRaisesRegex(RuntimeError,'bound_artifact_SHA_or_bytes_mismatch'):self.inspect()

    def test_each_current_consumer_is_full_SHA_bound(self):
        for name in (audit.EPISODE_SOURCE,audit.PREDICTION_SOURCE):
            body=(self.root/name).read_bytes()
            with self.subTest(name=name):
                self.write(name,body+b'\n# unexpected\n')
                with self.assertRaisesRegex(RuntimeError,'bound_artifact_SHA_or_bytes_mismatch'):self.inspect()
            self.write(name,body)

    def test_each_original_consumer_snapshot_is_full_SHA_bound(self):
        for name in (EPISODE_SNAPSHOT,PREDICTION_SNAPSHOT):
            body=(self.root/name).read_bytes()
            with self.subTest(name=name):
                self.write(name,body+b'\n# changed history\n')
                with self.assertRaisesRegex(RuntimeError,'bound_artifact_SHA_or_bytes_mismatch'):self.inspect()
            self.write(name,body)

    def test_wrong_historical_consumer_identity_is_rejected(self):
        self.history['prediction_store_history']['historical_source_bytes']+=1
        with self.assertRaisesRegex(RuntimeError,'exact_two_descriptor_consumer_histories_required'):self.inspect()

    def test_old_snapshot_cannot_replace_current_canonical_runtime(self):
        body=(self.root/EPISODE_SNAPSHOT).read_bytes()
        self.history['current_implementation']=self.write(audit.EPISODE_SOURCE,body)
        with self.assertRaisesRegex(RuntimeError,'descriptor_consumer_original_and_current_binding_mismatch'):self.inspect()

    def test_contract_and_actual_acceptance_descriptors_are_bound(self):
        for key in ('software_fix_contract','actual_new_software_acceptance'):
            path=self.history[key]['path'];body=(self.root/path).read_bytes()
            with self.subTest(key=key):
                self.write(path,body+b' ')
                with self.assertRaisesRegex(RuntimeError,'bound_artifact_SHA_or_bytes_mismatch'):self.inspect()
            self.write(path,body)

    def test_previous_verifier_remains_byte_exact(self):
        path=self.root/audit.PREVIOUS_VERIFIER;path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(RuntimeError,'pinned_verifier_or_amendment_changed'):self.inspect()

    def test_unclosed_W3_cannot_read_any_episode_or_write_audit(self):
        self.write('status/tasks.json',{'W1':{'status':'COMPLETE'},'W3':{'status':'RUNNING'}})
        with patch.object(audit,'load_episode_catalog',side_effect=AssertionError('no episode access')) as loader:
            with redirect_stdout(io.StringIO()):result=audit.main(['--root',str(self.root),'--execute'])
        self.assertEqual(result,2);loader.assert_not_called()
        self.assertFalse((self.root/audit.OUTPUT).exists())

    def test_preexisting_new_audit_directory_is_never_overwritten(self):
        self.write('status/tasks.json',{'W1':{'status':'COMPLETE'},'W3':{'status':'COMPLETE'}})
        (self.root/audit.OUTPUT).mkdir(parents=True)
        with patch.object(audit,'software_integrity',side_effect=AssertionError('must reject output first')):
            with self.assertRaisesRegex(RuntimeError,'preexisting_audit_preserved_no_overwrite'):
                audit.main(['--root',str(self.root),'--execute'])


if __name__=='__main__':unittest.main()
