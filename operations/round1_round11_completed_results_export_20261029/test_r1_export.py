import contextlib
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import r1_export as export


class FakeGitHub:
    """In-memory transport for byte restoration tests; never uses a network."""
    def __init__(self):self.contents={};self.assets={};self.uploads=0
    def ensure_release(self,*args):return {'id':1}
    def upload_part(self,release,path,name,checksum):
        content=Path(path).read_bytes()
        assert hashlib.sha256(content).hexdigest()==checksum
        self.uploads+=1;url='memory://'+name;self.assets[url]=content
        return dict(id=self.uploads,url=url)
    def git_publish(self,files,message):
        self.contents.update(files)
        return hashlib.sha1(message.encode()).hexdigest()


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.root=self.base/'project';self.root.mkdir();self.work=self.base/'upload'
        self.args=SimpleNamespace(root=str(self.root),work=str(self.work),credentials=None)
        self.stdout=contextlib.redirect_stdout(io.StringIO());self.stdout.__enter__()
    def tearDown(self):self.stdout.__exit__(None,None,None);self.temp.cleanup()
    def write(self,path,value):
        path=self.root/path;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value if isinstance(value,bytes) else export.encoded(value));return path
    def receipt(self,path,status,**values):return self.write(path,dict(status=status,synthetic=False,**values))
    def complete_development(self,key='dev'):
        base=export.CAMPAIGN+'/development/'+key
        self.receipt(base+'/receipt.json','DEVELOPMENT_EVALUATION_COMPLETE',scores=[.51,.82])
        self.write(base+'/fitted/parameters.pt',b'\0\xffweights')
        self.write(base+'/predictions.npz',b'PK\x03\x04\xffprediction bytes')
        self.write(base+'/model.pkl',b'\x80\x05\0model')
        self.write(base+'/metrics.csv',b'method,metric\r\nP,0.82\r\n')
        return base

    def test_native_receipts_control_scope_and_all_scientific_formats(self):
        base=self.complete_development()
        self.write(export.CAMPAIGN+'/development/test-double/receipt.json',
            dict(status='DEVELOPMENT_EVALUATION_COMPLETE',synthetic=True))
        self.receipt(export.CAMPAIGN+'/development/unfinished/receipt.json','PAUSED')
        self.write(export.OPERATIONAL+'/progress.json',dict(status='COMPLETE',tasks=[]))
        self.write(base+'/cache/raw.npy',b'raw input')
        self.write(base+'/credentials/github.json',b'private')
        external=self.base/'outside';external.write_bytes(b'private external')
        (self.root/base/'outside-link').symlink_to(external)
        bundle=export.inventory(self.root,self.work)
        self.assertEqual(bundle['completed_native_receipts'],1)
        self.assertFalse(bundle['whole_round_completed'])
        self.assertEqual(len(bundle['groups']),1)
        paths={p.relative_to(self.root).as_posix() for p in export._group_files(self.root,bundle['groups'][0])}
        self.assertIn(base+'/model.pkl',paths);self.assertIn(base+'/predictions.npz',paths)
        self.assertIn(base+'/fitted/parameters.pt',paths)
        self.assertFalse(any('/cache/' in p or '/credentials/' in p or 'outside-link' in p for p in paths))

    def test_source_models_and_partial_calibration_have_their_own_completed_seals(self):
        self.receipt('runs/source_training_v2/source_profile_development.json',
            'SOURCE_PROFILE_NEURAL_COMPLETE_CLASSICAL_PENDING')
        self.receipt('runs/source_training_v2/final-seed/status.json','SOURCE_FIT_COMPLETE')
        self.write('runs/source_training_v2/final-seed/selected.pt',b'actual source model')
        self.receipt('runs/source_classical_v2/receipt.json','SOURCE_CLASSICAL_CORE_COMPLETE')
        self.write('runs/source_classical_v2/lda/model.npz',b'actual classical model')
        candidate=export.CAMPAIGN+'/profiles/B5/fits/job/attempt-000001'
        self.receipt(candidate+'/candidate_state.json','FIT_PARAMETERS_COMMITTED')
        self.write(candidate+'/fitted/parameters.pt',b'completed B5 fit')
        self.write(export.CAMPAIGN+'/profiles/B5/progress.json',dict(completed={'one':{}},active=None))
        bundle=export.inventory(self.root,self.work)
        bases={g['base'] for g in bundle['groups']}
        self.assertEqual(bases,{'runs/source_training_v2','runs/source_classical_v2',candidate})
        self.assertFalse(bundle['whole_round_completed'])
        self.assertEqual(bundle['completed_native_receipts'],4)

    def test_extension_catalog_follows_owned_f0_and_refuses_wrong_descriptor(self):
        arm=export.ARMS[0];base='runs/source_extensions/'+arm+'/seed-20261018'
        receipt=self.receipt(base+'/f0_receipt.json','SOURCE_EXTENSION_F0_COMPLETE',arm=arm)
        self.write(base+'/fit/selected.pt',b'extension source model')
        downstream='runs/source_extension_downstream/'+arm
        descriptor=dict(path=receipt.relative_to(self.root).as_posix(),sha256=export.digest_file(receipt))
        self.write(downstream+'/catalog.json',dict(arm=arm,source_f0={'20261018':descriptor}))
        bundle=export.inventory(self.root,self.work)
        self.assertIn(base,{g['base'] for g in bundle['groups']})
        descriptor['sha256']='0'*64
        self.write(downstream+'/catalog.json',dict(arm=arm,source_f0={'20261018':descriptor}))
        changed=export.inventory(self.root,self.work)
        self.assertNotIn(base,{g['base'] for g in changed['groups']})
        self.assertTrue(any(r['error_type']=='NATIVE_CATALOG_RECEIPT_SHA256_DIFFERS' for r in changed['warnings']))

    def test_progress_heartbeat_preserves_frozen_metadata_until_completed_frontier_advances(self):
        self.complete_development('one')
        progress=self.write(export.CAMPAIGN+'/campaign_progress.json',dict(utc='first'))
        before=export.inventory(self.root,self.work)
        progress.write_bytes(export.encoded(dict(utc='second')))
        after=export.inventory(self.root,self.work)
        self.assertEqual(before['inventory_id'],after['inventory_id'])
        frozen=self.work/'r1-inventory'/'metadata'/after['metadata_epoch']/export.CAMPAIGN/'campaign_progress.json'
        self.assertEqual(json.loads(frozen.read_bytes()),dict(utc='first'))
        self.complete_development('two')
        advanced=export.inventory(self.root,self.work)
        self.assertNotEqual(after['inventory_id'],advanced['inventory_id'])
        self.assertEqual(json.loads((self.work/'r1-inventory'/'metadata'/advanced['metadata_epoch']/export.CAMPAIGN/'campaign_progress.json').read_bytes()),dict(utc='second'))

    def test_receipt_projection_cache_does_not_reparse_unchanged_large_payload(self):
        import receipt_projection
        path=self.receipt(export.CAMPAIGN+'/development/large/receipt.json',
            'DEVELOPMENT_EVALUATION_COMPLETE',payload='scientific output '*180000)
        first=export.inventory(self.root,self.work)
        self.assertEqual(first['completed_native_receipts'],1)
        with patch.object(receipt_projection,'read_projected',side_effect=AssertionError('large receipt reloaded')):
            self.assertEqual(export.inventory(self.root,self.work)['inventory_id'],first['inventory_id'])
        path.write_bytes(path.read_bytes()[:-3])
        broken=export.inventory(self.root,self.work)
        self.assertEqual(broken['completed_native_receipts'],0)
        self.assertTrue(broken['warnings'])

    def test_paper_preserves_complete_receipt_and_large_utf8_report_bytes_without_recomputation(self):
        base=self.complete_development();report=(('科学结果\r\n'*160000)+'end\n').encode()
        self.write(export.CAMPAIGN+'/reports/full.json',report)
        captured={}
        def publish(gh,round_number,files,work,metadata=None):
            for value in files:
                captured[value['source_path']]=b''.join(Path(part['local_path']).read_bytes() for part in value['parts'])
            metadata_record.update(metadata)
            return dict(status='PAPER_DATA_FILES_VERIFIED')
        metadata_record={}
        with patch.object(export,'_client',return_value=object()),patch('transport.lossless_publish.publish_snapshot',side_effect=publish):
            result=export.paper_once(self.args)
        self.assertTrue(result['paper_ready']);self.assertFalse(result['whole_round_completed'])
        self.assertEqual(captured[export.CAMPAIGN+'/reports/full.json'],report)
        self.assertEqual(captured[base+'/receipt.json'],(self.root/base/'receipt.json').read_bytes())
        self.assertEqual(captured[base+'/metrics.csv'],b'method,metric\r\nP,0.82\r\n')
        self.assertFalse(metadata_record['report_recomputed'])
        self.assertFalse(any(p.endswith('.pt') for p in captured))

    def test_actual_archive_restores_all_original_bytes_resume_never_changes_sources(self):
        self.complete_development()
        self.receipt('runs/internal_validation_once/candidates/committed/predictions/receipt.json','PREDICTIONS_COMMITTED')
        self.write('runs/internal_validation_once/candidates/committed/fitted/parameters.json',b'{"parameter":3.14159}\n')
        self.write('runs/internal_validation_once/candidates/uncommitted/fitted/parameters.json',b'{"unfinished":true}\n')
        original={p.relative_to(self.root).as_posix():p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        gh=FakeGitHub();statuses=[]
        with patch.object(export,'_client',return_value=gh):
            for i in range(20):
                result=export.full_once(self.args);statuses.append(result)
                if result.get('complete'):break
            self.assertTrue(result['complete']);self.assertFalse(result['whole_round_completed'])
            uploaded=gh.uploads
            self.assertTrue(export.full_once(self.args)['complete'])
            self.assertEqual(gh.uploads,uploaded)
        pointer=json.loads(gh.contents[export.PREFIX+'/COMPLETED_OUTPUT_LATEST.json'])
        manifest=json.loads(gh.contents[pointer['manifest']]);restored={}
        for group in manifest['groups']:
            archived=json.loads(gh.contents[group['manifest']])
            content=b''.join(gh.assets[p['url']] for p in archived['parts'])
            with tarfile.open(fileobj=io.BytesIO(content),mode='r:gz') as archive:
                members={m.name:archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
            for item in archived['files']:
                data=members[item['source_path']]
                self.assertEqual(len(data),item['bytes'])
                self.assertEqual(hashlib.sha256(data).hexdigest(),item['sha256'])
                restored[item['source_path']]=data
        self.assertTrue(any(p.endswith('parameters.pt') for p in restored))
        self.assertTrue(any('/committed/fitted/parameters.json' in p for p in restored))
        self.assertFalse(any('/uncommitted/' in p for p in restored))
        self.assertEqual(original,{p.relative_to(self.root).as_posix():p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        self.assertTrue(all(data==original[name] for name,data in restored.items()))

    def test_whole_scientific_completion_requires_internal_and_both_original_extension_arms(self):
        self.receipt(export.CAMPAIGN+'/pre_internal_freeze.json','IMMUTABLE_PRE_INTERNAL_DEVELOPMENT_FREEZE')
        self.receipt('runs/internal_validation_once/receipt.json','ONE_TIME_INTERNAL_CHECK_COMPLETE')
        self.assertFalse(export.inventory(self.root,self.work)['whole_round_completed'])
        for arm in export.ARMS:
            self.receipt('runs/source_extension_downstream/'+arm+'/pre_internal_development_freeze.json',
                'SOURCE_EXTENSION_PRE_INTERNAL_DEVELOPMENT_FROZEN',arm=arm)
        self.assertTrue(export.inventory(self.root,self.work)['whole_round_completed'])

    def test_new_nested_result_advances_group_without_reloading_old_receipt(self):
        import receipt_projection
        base=self.complete_development()
        before=export.inventory(self.root,self.work)
        self.write(base+'/fitted/later-record.json',dict(actual='new committed output'))
        with patch.object(receipt_projection,'read_projected',side_effect=AssertionError('unchanged receipt reloaded')):
            after=export.inventory(self.root,self.work)
        self.assertNotEqual(before['groups'][0]['identity'],after['groups'][0]['identity'])
        self.assertNotEqual(before['inventory_id'],after['inventory_id'])

    def test_declared_completed_but_missing_receipt_is_disclosed_and_never_claims_full_upload(self):
        self.complete_development()
        missing=export.CAMPAIGN+'/development/missing/receipt.json'
        self.write(export.OPERATIONAL+'/progress.json',dict(tasks=[dict(status='COMPLETE',expected_receipt=missing)]))
        bundle=export.inventory(self.root,self.work)
        self.assertEqual(bundle['warnings'],[dict(path=missing,error_type='DECLARED_COMPLETED_RECEIPT_MISSING')])
        gh=FakeGitHub()
        with patch.object(export,'_client',return_value=gh):
            for _ in range(10):
                result=export.full_once(self.args)
                if result['status']=='AVAILABLE_OUTPUT_UPLOADED_WITH_RECEIPT_GAPS':break
        self.assertFalse(result['complete']);self.assertEqual(result['pending_receipt_errors'],1)
        pointer=json.loads(gh.contents[export.PREFIX+'/COMPLETED_OUTPUT_LATEST.json'])
        self.assertEqual(pointer['unresolved_receipts'],bundle['warnings'])

    def test_malformed_extension_catalog_cannot_block_other_completed_data_or_claim_full_scope(self):
        self.complete_development()
        self.write('runs/source_extension_downstream/'+export.ARMS[0]+'/catalog.json',b'{"truncated":')
        bundle=export.inventory(self.root,self.work)
        self.assertEqual(bundle['completed_native_receipts'],1)
        self.assertTrue(any(row['error_type'].startswith('EXTENSION_CATALOG_') for row in bundle['warnings']))


if __name__=='__main__':unittest.main()
