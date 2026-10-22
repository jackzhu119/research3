"""Fail-closed software, transfer conservation and recovered W3 audit gates.

Fixtures are tiny synthetic bookkeeping objects, never scientific samples or
claims of completed W3. Pure reconciliation itself is covered by the frozen
deployment-builder tests; these isolate the new closed-completion bindings.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

from tools import audit_completed_stieger_w3 as audit
from tools import build_data_plan_deploy_manifest as deploy

REPO=Path(__file__).resolve().parents[1]
RECOVERY='audit/data_plan_20261020/runtime_recovery/d75c8fa0-1d33-45dd-b8d9-78f3e0dfbb34'
SOURCE_TRAINING_SNAPSHOT='hpc/fixtures/original_source_training_54b972.py.txt'
SOURCE_TRAINING_HISTORY={'path':'p3work/source_training.py',
    'sha256':'54b972457ea1bbc08c15aeff9440a418047b7752b725dac3c27d562bead2a849','bytes':80473}
PILOT_SNAPSHOT='hpc/fixtures/original_run_controlled_pilot_856393.py.txt'
PILOT_HISTORY={'path':'tools/run_controlled_pilot.py',
    'sha256':'856393030d10cdbbee54138fefadce6ee8520243f9e54d00feb7d5328088172c','bytes':50853}


def historical_source_training(manifest, snapshot=None, *, expected=SOURCE_TRAINING_HISTORY):
    """This unit fixture exercises the frozen verifier, not the HPC runtime."""
    historical=[row for row in manifest['files'] if row['path']==expected['path']]
    if historical != [expected]:
        raise AssertionError('exact_original_source_training_inventory_required')
    body=(snapshot or REPO/SOURCE_TRAINING_SNAPSHOT).read_bytes()
    actual={'path':expected['path'],'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body)}
    if actual != expected:
        raise AssertionError('historical_source_training_snapshot_SHA_or_bytes_mismatch')
    return body


class AuditCompletedStiegerTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        manifest=json.loads((REPO/'audit/software_code_manifest.json').read_bytes())
        paths={r['path'] for r in manifest['files']}
        paths.update({'audit/software_code_manifest.json',audit.HISTORICAL_VERIFIER,audit.PRECISION_AMENDMENT,
            audit.FROZEN_CLOSURE,deploy.RECOVERY_TOOL,'audit/data_plan_20261019/stieger/cache_plan.json',
            RECOVERY+'/runtime_reconciliation_amendment.json',RECOVERY+'/transaction.json'})
        paths.update(deploy.PRECISION_AUDIT+'/'+p for p in
            ('original_classical.py','original_software_code_manifest.json','native_regression.json'))
        for relative in paths:
            target=self.root/relative;target.parent.mkdir(parents=True,exist_ok=True)
            source=REPO/relative
            if relative=='p3work/source_training.py':
                # Use the exact original bytes expected by the historical 0f1
                # verifier; the selected school runtime has separate checks.
                target.write_bytes(historical_source_training(manifest))
                continue
            if relative=='tools/run_controlled_pilot.py':
                target.write_bytes(historical_source_training(manifest,REPO/PILOT_SNAPSHOT,expected=PILOT_HISTORY))
                continue
            if relative=='p3work/episodes.py':
                # This fixture exercises the unchanged historical 0f1 verifier.
                # The new canonical compatibility runtime is tested separately.
                import hashlib
                historical=[r for r in manifest['files'] if r['path']==relative]
                expected={'path':relative,'sha256':'348604283251b8d50a34c5b0545a895afb7bbaa024ff82d7c62be11e30504a65','bytes':43688}
                self.assertEqual(historical,[expected])
                source=REPO/'audit/data_plan_20261020/episode_descriptor_compatibility_fix/original_episodes_348604.py.txt'
                body=source.read_bytes()
                self.assertEqual({'path':relative,'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body)},expected)
            elif relative=='p3work/prediction_store.py':
                import hashlib
                historical=[r for r in manifest['files'] if r['path']==relative]
                expected={'path':relative,'sha256':'e1af634b84bc234bc71394c6fff222d4cd67b2d4ce7f88bc2da5a40b4abde5ab','bytes':32085}
                self.assertEqual(historical,[expected])
                source=REPO/'audit/data_plan_20261020/episode_descriptor_compatibility_fix/original_prediction_store_e1af63.py.txt'
                body=source.read_bytes()
                self.assertEqual({'path':relative,'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body)},expected)
            shutil.copyfile(source,target)
        self.plan=json.loads((self.root/'audit/data_plan_20261019/stieger/cache_plan.json').read_bytes())

    def write(self,path,value):
        target=self.root/path;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(value if isinstance(value,bytes) else json.dumps(value).encode())
        return deploy.descriptor(self.root,path)

    def receipt(self,path,entries):
        prefix='paper3/unit-stieger-completion/'
        objects=[]
        for item,role in entries:
            objects.append({**item,'role':role,'dataset':'Stieger2021','key':prefix+item['path'],
                'status':'verified','put':{'status':'created','http_status':200,
                    'request_body_bytes':item['bytes'],'request_body_sha256':item['sha256']},
                'get':{'status':'verified','http_status':200,'complete_body_read':True,
                    'bytes_read':item['bytes'],'sha256':item['sha256'],'declared_content_length':item['bytes']}})
        return self.write(path,{'schema_version':1,'operation':'paper3_explicit_manifest_backup',
            'status':'verified','execute':True,'verification':'full_get_body_sha256','prefix':prefix,'objects':objects})

    def state(self):
        artifacts=[self.write('cache/stieger_v2/unit/'+name,b'synthetic bookkeeping bytes')
                   for name in ('X.npy','events.json','metadata.json')]
        receipt=self.receipt('cache/stieger_v2/unit/cache_receipt.json',[(a,
            'audit' if Path(a['path']).name=='metadata.json' else 'result') for a in artifacts])
        return {'file_id':1,'raw_path':'data/never_opened.mat','artifacts':artifacts,
            'cache_backup_receipt':receipt,'cache_bytes':sum(a['bytes'] for a in artifacts),
            'official_provider_body_bytes':10,'official_provider_verified_body_bytes':10,
            'r2_restore_body_bytes':15,'r2_restore_verified_body_bytes':0,
            'transfer_history':[
                {'kind':'official_get','file_id':1,'raw_path':'data/never_opened.mat','reserved_bytes':20,
                 'status':'SUCCESS_ACTUAL_BODY_SETTLED','actual_verified_bytes':10},
                {'kind':'r2_restore','file_id':1,'raw_path':'data/never_opened.mat','reserved_bytes':15,
                 'status':'HARD_INTERRUPTION_FULL_RESERVATION_CONSUMED'}]}

    def completed(self):
        self.states=[self.state()]*598
        self.manifest_path=self.root/'cache/stieger_v2/stieger_cache.json'
        self.manifest=self.write('cache/stieger_v2/stieger_cache.json',{'fixture':'no scientific cache'})
        self.amendment=deploy.descriptor(self.root,RECOVERY+'/runtime_reconciliation_amendment.json')
        value=json.loads((self.root/self.amendment['path']).read_bytes())
        self.builder=value['implementation'];self.cap=value['effective_Stieger_CPU_wall_cap_seconds']
        floor=value['conservative_accounting']['new_wall_high_water_seconds']['Stieger2021']
        total={k:self.states[0][k]*598 for k in
               ('official_provider_body_bytes','r2_restore_body_bytes','cache_bytes')}
        self.task={'cache_manifest':self.manifest,'cache_r2_receipt':{'path':'audit/global.json','sha256':'a'*64},
            'cpu_operational_recovery':self.amendment,'effective_CPU_wall_cap_seconds':self.cap}
        self.completion={'schema_version':'paper3-recovered-Stieger-CPU-completion-v1',
            'amendment':self.amendment,'operational_builder':self.builder,
            'scientific_plan_sha256':self.plan['plan_sha256'],'cache_manifest':self.manifest,
            'original_global_cache_GET_receipt':self.task['cache_r2_receipt'],
            'effective_CPU_wall_cap_seconds':self.cap,'original_plan_wall_cap_seconds':self.plan['max_wall_seconds'],
            'original_body_caps_unchanged':True,'cumulative_wall_seconds_before_operational_backup':floor+1,
            'official_provider_body_bytes':total['official_provider_body_bytes'],
            'r2_restore_body_bytes':total['r2_restore_body_bytes'],
            'scientific_fits':0,'GPU_actions':0,'remote_server_actions':False,'external_data_access':False}
        self.progress={'status':'W3_COMPLETE_CPU_CACHE_AND_FULL_GET_ONLY','completed_sessions':598,
            'plan_sha256':self.plan['plan_sha256'],'cache_manifest':self.manifest,
            'cache_r2_receipt':self.task['cache_r2_receipt'],'explicit_runtime_reconciliation':self.amendment,
            'effective_CPU_wall_cap_seconds':self.cap,'original_plan_wall_cap_seconds':self.plan['max_wall_seconds'],
            'operational_builder':self.builder,'active_wall_elapsed_s':floor+2,
            'gpu_jobs_started':0,'remote_jobs_started':0,'external_payload_accessed':False,**total}
        self.publish_completion()

    def publish_completion(self):
        completion=self.write('cache/stieger_v2/cpu_recovery_completion.json',self.completion)
        receipt=self.receipt('cache/stieger_v2/cpu_recovery_backup_receipt.json',
            [(completion,'audit'),(self.amendment,'audit'),(self.builder,'code')])
        for value in (self.task,self.progress):
            value.update(cpu_operational_completion=completion,cpu_operational_R2_receipt=receipt)
        self.publish_progress()

    def publish_progress(self):
        self.write('cache/stieger_v2/progress.json',self.progress)

    def inspect_completion(self):
        def isolated_verified_reconciliation(closure,path):
            # Only the extra operational-completion checks are under test here.
            # Frozen builder tests validate actual pure reconciliation/caps.
            closure.runtime_recovery_history.append({'amendment':self.amendment,
                'transaction':deploy.descriptor(self.root,path)})
        with patch.object(deploy.Closure,'load_runtime_recovery',isolated_verified_reconciliation):
            return audit.cpu_completion(self.root,self.plan,self.task,self.manifest_path,self.states)

    def test_exact_116_and_one_approved_precision_change_pass(self):
        report=audit.software_integrity(self.root)
        self.assertEqual(report['historical_unchanged_files'],116)
        self.assertEqual(report['approved_modified_files'],1)
        self.assertNotIn('old117_code_SHA_unchanged',report)

    def test_historical_source_training_snapshot_tampering_is_rejected_before_copy(self):
        manifest=json.loads((REPO/'audit/software_code_manifest.json').read_bytes())
        snapshot=self.root/'tampered-original-source-training.txt'
        snapshot.write_bytes((REPO/SOURCE_TRAINING_SNAPSHOT).read_bytes()+b'\n# changed\n')
        with self.assertRaisesRegex(AssertionError,'snapshot_SHA_or_bytes_mismatch'):
            historical_source_training(manifest,snapshot)

    def test_historical_source_training_inventory_is_not_replaced_by_current_runtime(self):
        manifest=json.loads((REPO/'audit/software_code_manifest.json').read_bytes())
        row=next(row for row in manifest['files'] if row['path']=='p3work/source_training.py')
        row['sha256']='0'*64
        with self.assertRaisesRegex(AssertionError,'exact_original_source_training_inventory'):
            historical_source_training(manifest)

    def test_historical_pilot_archive_tampering_is_rejected_before_copy(self):
        manifest=json.loads((REPO/'audit/software_code_manifest.json').read_bytes())
        snapshot=self.root/'tampered-original-pilot.txt'
        snapshot.write_bytes((REPO/PILOT_SNAPSHOT).read_bytes()+b'\n# changed\n')
        with self.assertRaisesRegex(AssertionError,'snapshot_SHA_or_bytes_mismatch'):
            historical_source_training(manifest,snapshot,expected=PILOT_HISTORY)

    def test_unapproved_second_original_file_change_rejected(self):
        self.write('p3work/__init__.py',b'changed\n')
        with self.assertRaisesRegex(RuntimeError,'bound_artifact_SHA_or_bytes_mismatch'):
            audit.software_integrity(self.root)

    def test_current_precision_implementation_tampering_rejected(self):
        self.write(deploy.PRECISION_SOURCE,b'changed\n')
        with self.assertRaises(deploy.DeploymentError):audit.software_integrity(self.root)

    def test_archived_original_precision_tampering_rejected(self):
        self.write(deploy.PRECISION_AUDIT+'/original_classical.py',b'changed\n')
        with self.assertRaises(deploy.DeploymentError):audit.software_integrity(self.root)

    def test_amendment_historical_checker_and_old_inventory_are_pinned(self):
        for path in (audit.PRECISION_AMENDMENT,audit.HISTORICAL_VERIFIER,'audit/software_code_manifest.json'):
            before=(self.root/path).read_bytes()
            with self.subTest(path=path):
                self.write(path,before+b' ')
                with self.assertRaisesRegex(RuntimeError,'pinned_verifier_or_amendment_changed'):
                    audit.software_integrity(self.root)
            self.write(path,before)

    def test_closed_transfer_keeps_failed_entire_reservation_charged(self):
        result=audit.session_accounting(self.root,self.state())
        self.assertEqual(result['official_provider_body_bytes'],10)
        self.assertEqual(result['r2_restore_body_bytes'],15)

    def test_open_intent_missing_history_refund_and_wrong_success_rejected(self):
        for change in ('intent','missing_history','refunded_failure','over_reserve','unknown_status','changed_identity'):
            with self.subTest(change=change):
                state=self.state()
                if change=='intent':state['transfer_intent']={}
                if change=='missing_history':state.pop('transfer_history')
                if change=='refunded_failure':state['r2_restore_body_bytes']=0
                if change=='over_reserve':state['transfer_history'][0]['actual_verified_bytes']=21
                if change=='unknown_status':state['transfer_history'][1]['status']='IGNORED'
                if change=='changed_identity':state['transfer_history'][1]['file_id']=2
                with self.assertRaises(RuntimeError):audit.session_accounting(self.root,state)

    def test_artifact_size_or_receipt_role_and_partial_GET_rejected(self):
        for change in ('cache_size','artifact_bytes','receipt_role','partial_GET'):
            with self.subTest(change=change):
                state=self.state()
                if change=='cache_size':state['cache_bytes']-=1
                if change=='artifact_bytes':state['artifacts'][0]['bytes']-=1
                if change in ('receipt_role','partial_GET'):
                    value=json.loads((self.root/state['cache_backup_receipt']['path']).read_bytes())
                    if change=='receipt_role':value['objects'][0]['role']='audit'
                    else:value['objects'][0]['get']['complete_body_read']=False
                    state['cache_backup_receipt']=self.write(state['cache_backup_receipt']['path'],value)
                with self.assertRaises(Exception):audit.session_accounting(self.root,state)

    def test_closed_recovered_completion_is_bound_and_full_GET_exact(self):
        self.completed();result=self.inspect_completion()
        self.assertEqual(result['operational_R2_objects'],3)
        self.assertTrue(result['original_body_caps_unchanged'])

    def test_recovery_helper_is_mandatory_and_errors_fail_closed(self):
        self.completed()
        with patch.object(deploy.Closure,'load_runtime_recovery',side_effect=deploy.DeploymentError('bad_reconciliation')):
            with self.assertRaisesRegex(deploy.DeploymentError,'bad_reconciliation'):
                audit.cpu_completion(self.root,self.plan,self.task,self.manifest_path,self.states)

    def test_closed_progress_open_clock_GPU_remote_external_or_partial_rejected(self):
        self.completed()
        for key,value in (('wall_intent',{}),('gpu_jobs_started',1),('remote_jobs_started',1),
                          ('external_payload_accessed',True),('completed_sessions',597),('status','RUNNING')):
            old=copy.deepcopy(self.progress)
            with self.subTest(key=key):
                self.progress[key]=value;self.publish_progress()
                with self.assertRaises(RuntimeError):self.inspect_completion()
            self.progress=old

    def test_elapsed_floor_strict_cap_and_nonfinite_values_rejected(self):
        self.completed()
        for value in (0,self.cap,float('inf'),float('nan'),True):
            with self.subTest(value=value):
                self.progress['active_wall_elapsed_s']=value;self.publish_progress()
                with self.assertRaises(RuntimeError):self.inspect_completion()

    def test_completion_before_backup_cannot_exceed_final_elapsed_or_refund_gap(self):
        self.completed()
        for value in (0,self.progress['active_wall_elapsed_s']+1):
            with self.subTest(value=value):
                self.completion['cumulative_wall_seconds_before_operational_backup']=value
                self.publish_completion()
                with self.assertRaises(RuntimeError):self.inspect_completion()

    def test_original_caps_and_bodies_cannot_be_rewritten(self):
        self.completed()
        for target,key,value in ((self.task,'effective_CPU_wall_cap_seconds',self.cap+1),
                (self.progress,'original_plan_wall_cap_seconds',1),
                (self.progress,'r2_restore_body_bytes',0),
                (self.completion,'official_provider_body_bytes',0),
                (self.completion,'original_body_caps_unchanged',False)):
            previous=copy.deepcopy(target)
            with self.subTest(key=key):
                target[key]=value;self.publish_completion()
                with self.assertRaises(RuntimeError):self.inspect_completion()
            target.clear();target.update(previous)

    def test_operational_receipt_requires_exact_three_actual_full_GETs(self):
        self.completed()
        for change in ('missing','extra','role','partial'):
            self.publish_completion()
            path=self.task['cpu_operational_R2_receipt']['path']
            value=json.loads((self.root/path).read_bytes())
            if change=='missing':value['objects'].pop()
            if change=='extra':
                extra=self.write('audit/extra.json',{})
                extra_receipt=self.receipt('audit/extra_receipt.json',[(extra,'audit')])
                value['objects']+=json.loads((self.root/extra_receipt['path']).read_bytes())['objects']
            if change=='role':value['objects'][2]['role']='audit'
            if change=='partial':value['objects'][0]['get']['complete_body_read']=False
            receipt=self.write(path,value)
            self.task['cpu_operational_R2_receipt']=receipt;self.progress['cpu_operational_R2_receipt']=receipt
            self.publish_progress()
            with self.subTest(change=change):
                with self.assertRaises(Exception):self.inspect_completion()

    def test_missing_or_different_reconciliation_marker_rejected(self):
        self.completed();self.progress.pop('explicit_runtime_reconciliation');self.publish_progress()
        with self.assertRaisesRegex(RuntimeError,'explicit_CPU_reconciliation'):
            self.inspect_completion()

    def test_stale_global_receipt_or_manifest_binding_rejected(self):
        self.completed()
        for key,value in (('original_global_cache_GET_receipt',{'path':'audit/other.json','sha256':'b'*64}),
                          ('cache_manifest',{'path':'cache/wrong.json','sha256':'c'*64})):
            previous=copy.deepcopy(self.completion)
            with self.subTest(key=key):
                self.completion[key]=value;self.publish_completion()
                with self.assertRaises(RuntimeError):self.inspect_completion()
            self.completion=previous

    def test_pending_live_state_does_not_load_episodes_or_write_report(self):
        self.write('status/tasks.json',{'W1':{'status':'COMPLETE'},'W3':{'status':'IN_PROGRESS_LOCAL_CPU'}})
        for execute in (False,True):
            stream=io.StringIO()
            with (patch.object(audit,'load_episode_catalog',side_effect=AssertionError('episode access forbidden before COMPLETE')),
                  redirect_stdout(stream)):
                result=audit.main(['--root',str(self.root)]+(['--execute'] if execute else []))
            self.assertEqual(result,2 if execute else 0)
            output=json.loads(stream.getvalue());self.assertTrue(output['status'].startswith('PENDING_'))
            self.assertFalse(output['completion_report_written']);self.assertFalse((self.root/audit.OUTPUT).exists())

    def test_symlink_output_rejected_before_artifact_or_episode_audit(self):
        self.write('status/tasks.json',{'W1':{'status':'COMPLETE'},'W3':{'status':'COMPLETE'}})
        output=self.root/audit.OUTPUT;output.parent.mkdir(parents=True,exist_ok=True)
        outside=self.root/'unrelated';outside.mkdir();output.symlink_to(outside,target_is_directory=True)
        with patch.object(audit,'software_integrity',side_effect=AssertionError('must reject path first')):
            with self.assertRaisesRegex(Exception,'symlink_forbidden'):
                audit.main(['--root',str(self.root),'--execute'])
        self.assertEqual(list(outside.iterdir()),[])


if __name__=='__main__':unittest.main()
