import io
import json
from pathlib import Path
import tarfile

import pytest

from paper3r10.platform import common,runtime,uploads
from test_platform_contract import complete,plan_fixture


def test_paper_collection_preserves_full_csv_large_json_and_real_coverage(tmp_path):
    path,plan=plan_fixture(tmp_path);runtime.prepare(path);complete(plan,plan['tasks'][0])
    reports=Path(plan['output'])/'reports';reports.mkdir()
    large={'status':'COMPLETE','recipe_sha256':plan['recipe_sha256'],'coverage':{'total_tasks':1,'sealed_complete_tasks':1,'complete_task_keys':[plan['tasks'][0]['key']],
             'complete_task_keys_sha256':common.digest([plan['tasks'][0]['key']])},'full_payload':'q'*(17*1024*1024)}
    common.atomic(reports/'report.json',large)
    csv=b'participant_id,ba,bce\r\n"S01",0.5,0.7\r\n'
    (reports/'participants.csv').write_bytes(csv)
    stage,paths,metadata=uploads.paper_collection(path,tmp_path/'work')
    assert metadata['collection_status']=='COMPLETE'
    assert (stage/'reports/report.json').read_bytes()==(reports/'report.json').read_bytes()
    assert (stage/'reports/participants.csv').read_bytes()==csv
    assert metadata['complete_tasks']==1 and metadata['required_missing']==[]


def test_stale_or_partial_report_cannot_claim_paper_complete(tmp_path):
    path,plan=plan_fixture(tmp_path);runtime.prepare(path);complete(plan,plan['tasks'][0])
    reports=Path(plan['output'])/'reports';reports.mkdir()
    common.atomic(reports/'report.json',{'status':'COMPLETE','recipe_sha256':'old-recipe'})
    _,_,metadata=uploads.paper_collection(path,tmp_path/'work')
    assert metadata['collection_status']=='PARTIAL'
    assert metadata['complete_tasks']==1 and metadata['required_missing']


def test_missing_participant_artifacts_are_explicit_partial_not_sanitized(tmp_path):
    path,plan=plan_fixture(tmp_path);runtime.prepare(path);complete(plan,plan['tasks'][0],paper=False)
    _,_,metadata=uploads.paper_collection(path,tmp_path/'work')
    assert plan['tasks'][0]['key']+':paper_artifacts' in metadata['required_missing']


def test_archive_parts_restore_all_bytes_and_resume_reuses_prepared_parts(tmp_path):
    base=tmp_path/'task';base.mkdir()
    (base/'predictions.bin').write_bytes(bytes(range(256))*30)
    (base/'state.json').write_bytes(b'{"synthetic":true,"rng":"original"}\n')
    folder,manifest=uploads.pack_group(base,list(base.iterdir()),tmp_path/'work','task-key','identity',part_bytes=111)
    parts=[folder/p['name'] for p in manifest['parts']]
    assert all(p.stat().st_size<=111 for p in parts)
    data=b''.join(p.read_bytes() for p in parts)
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
        assert archive.extractfile('predictions.bin').read()==(base/'predictions.bin').read_bytes()
        assert archive.extractfile('state.json').read()==(base/'state.json').read_bytes()
    stamps={p:p.stat().st_mtime_ns for p in parts}
    _,again=uploads.pack_group(base,list(base.iterdir()),tmp_path/'work','task-key','identity',part_bytes=111)
    assert manifest==again and all(p.stat().st_mtime_ns==s for p,s in stamps.items())
    assert all(common.sha(folder/p['name'])==p['sha256'] for p in manifest['parts'])


def test_changed_completed_source_rebuilds_archive_not_silent_reuse(tmp_path):
    base=tmp_path/'task';base.mkdir();p=base/'state.json';p.write_text('old')
    _,before=uploads.pack_group(base,[p],tmp_path/'work','key','identity',part_bytes=1024)
    p.write_text('new payload')
    _,after=uploads.pack_group(base,[p],tmp_path/'work','key','identity',part_bytes=1024)
    assert before['files'][0]['sha256']!=after['files'][0]['sha256']


def test_archive_rejects_symlink_and_private_sources(tmp_path):
    base=tmp_path/'task';base.mkdir();p=base/'credentials.json';p.write_text('not-a-real-token')
    with pytest.raises(ValueError):list(uploads.scientific_files(base))
    p.unlink();outside=tmp_path/'outside';outside.write_text('x');(base/'link').symlink_to(outside)
    with pytest.raises(ValueError):list(uploads.scientific_files(base))


def test_network_failure_never_changes_science_or_retries_training(tmp_path,monkeypatch):
    path,plan=plan_fixture(tmp_path);runtime.prepare(path);complete(plan,plan['tasks'][0])
    original={p:p.read_bytes() for p in common.task_folder(plan,plan['tasks'][0]).iterdir() if p.is_file()}
    monkeypatch.setattr(uploads,'discover_token',lambda *a:'fixture-token-not-real')
    def offline(*a,**k):raise ConnectionError('fixture network unavailable')
    result=uploads.channel_loop(path,'paper',gh_factory=offline,once=True)
    assert result['status']=='RETRYING' and result['training_controlled'] is False
    assert all(p.read_bytes()==v for p,v in original.items())
    assert common.verify_result(plan,plan['tasks'][0])


def test_upload_factory_gets_credential_but_logs_do_not_contain_it(tmp_path,monkeypatch,capsys):
    path,plan=plan_fixture(tmp_path);runtime.prepare(path)
    secret='fixture-secret-that-must-not-be-logged'
    monkeypatch.setattr(uploads,'discover_token',lambda *a:secret)
    def fail(token,*a,**k):
        assert token==secret
        raise RuntimeError('Exception body may contain '+secret)
    uploads.channel_loop(path,'paper',gh_factory=fail,once=True)
    assert secret not in capsys.readouterr().out
    assert secret not in (runtime.runtime_dir(plan)/'upload-paper/status.json').read_text()


def test_independent_channels_have_separate_state_and_remote_paths(tmp_path,monkeypatch):
    path,plan=plan_fixture(tmp_path);runtime.prepare(path)
    monkeypatch.setattr(uploads,'discover_token',lambda *a:'fixture')
    factory=lambda *a,**k:object()
    monkeypatch.setattr(uploads,'paper_cycle',lambda *a:dict(status='PAPER_DATA_FILES_VERIFIED',channel='paper'))
    monkeypatch.setattr(uploads,'full_cycle',lambda *a:dict(status='PARTIAL',channel='full'))
    paper=uploads.channel_loop(path,'paper',gh_factory=factory,once=True)
    full=uploads.channel_loop(path,'full',gh_factory=factory,once=True)
    assert paper['status']=='PAPER_DATA_FILES_VERIFIED' and full['status']=='PARTIAL'
    assert common.read(runtime.runtime_dir(plan)/'upload-paper/status.json')['status']=='PAPER_DATA_FILES_VERIFIED'
    assert common.read(runtime.runtime_dir(plan)/'upload-full/status.json')['status']=='PARTIAL'


def test_full_archive_get_rejects_partial_http_and_wrong_byte_count():
    class Response(io.BytesIO):
        status=206
    class Client:
        prefix_url='https://api.github.com/repos/fixture'
        def _request(self,*a,**k):return Response(b'partial')
    with pytest.raises(RuntimeError,match='HTTP200'):
        uploads.FullVerifiedGitHub._asset_hash(Client(),dict(id=1,size=7))


def test_reporter_failure_retries_without_credentials_or_scientific_changes(tmp_path,monkeypatch):
    from types import SimpleNamespace
    path,plan=plan_fixture(tmp_path);runtime.prepare(path);complete(plan,plan['tasks'][0]);calls=[]
    monkeypatch.setattr(uploads,'discover_token',lambda *_:(_ for _ in ()).throw(AssertionError('Reporter must not request credentials')))
    sealed=common.sha(common.task_folder(plan,plan['tasks'][0])/'completion.json')
    def fail(command,**kw):
        calls.append(command);assert kw['timeout']==300
        return SimpleNamespace(returncode=2)
    first=uploads.report_loop(path,runner=fail,once=True)
    assert first['status']=='RETRYING' and first['training_controlled'] is False
    def succeeds(command,**kw):
        calls.append(command);folder=Path(plan['output'])/'reports'
        common.atomic(folder/'report.json',{'status':'COMPLETE','recipe_sha256':plan['recipe_sha256']})
        common.atomic(folder/'report-status.json',{'status':'COMPLETE','recipe_sha256':plan['recipe_sha256'],
            'report_sha256':common.sha(folder/'report.json'),'total_tasks':1,'sealed_complete_tasks':1,
            'complete_task_keys_sha256':common.digest([plan['tasks'][0]['key']])})
        return SimpleNamespace(returncode=0)
    assert uploads.report_loop(path,runner=succeeds,once=True)['status']=='COMPLETE'
    assert len(calls)==2 and common.sha(common.task_folder(plan,plan['tasks'][0])/'completion.json')==sealed
    assert 'report' in calls[-1] and '--plan' in calls[-1]


def test_old_mechanism_upload_cannot_replace_formal_active_stage():
    mechanism={'stage':'mechanism','recipe_sha256':'a'*64}
    formal={'stage':'formal_exploratory','recipe_sha256':'b'*64}
    index=uploads.merge_stage_index({},mechanism,'paper',{'manifest':'a'})
    index=uploads.merge_stage_index(index,formal,'paper',{'manifest':'b'})
    again=uploads.merge_stage_index(index,mechanism,'full',{'manifest':'a-full'})
    assert again['active_stage']=='formal_exploratory' and again['active_recipe_sha256']=='b'*64
    assert again['stages']['mechanism']['a'*64]['channels']['full']['manifest']=='a-full'
    assert again['stages']['formal_exploratory']['b'*64]['channels']['paper']['manifest']=='b'
    assert uploads.stage_prefix(mechanism)!=uploads.stage_prefix(formal)
