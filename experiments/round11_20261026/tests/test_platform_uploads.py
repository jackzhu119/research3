import io
import json
import base64
import hashlib
from pathlib import Path
import tarfile

import pytest

from paper3r11.platform import common,runtime,uploads
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


def test_old_development_upload_cannot_replace_formal_active_stage():
    development={'stage':'development','recipe_sha256':'a'*64}
    formal={'stage':'exploratory','recipe_sha256':'b'*64}
    index=uploads.merge_stage_index({},development,'paper',{'manifest':'a'})
    index=uploads.merge_stage_index(index,formal,'paper',{'manifest':'b'})
    again=uploads.merge_stage_index(index,development,'full',{'manifest':'a-full'})
    assert again['active_stage']=='exploratory' and again['active_recipe_sha256']=='b'*64
    assert again['stages']['development']['a'*64]['channels']['full']['manifest']=='a-full'
    assert again['stages']['exploratory']['b'*64]['channels']['paper']['manifest']=='b'
    assert uploads.stage_prefix(development)!=uploads.stage_prefix(formal)


def test_progress_branch_created_even_if_local_status_read_temporarily_fails(tmp_path,monkeypatch):
    path,plan=plan_fixture(tmp_path);runtime.prepare(path);calls=[]
    class GitHub:
        branch='paper3-r11-progress'
        def ensure_branch(self):calls.append('ensure_branch');return 'a'*40
    monkeypatch.setattr(uploads,'status',lambda *_,**__:(_ for _ in ()).throw(ValueError('temporary local state')))
    with pytest.raises(ValueError):uploads.monitor_cycle(GitHub(),path,tmp_path/'monitor')
    assert calls==['ensure_branch']
    assert common.read(tmp_path/'monitor/branch-ready.json')['status']=='BRANCH_READY'


def _two_tasks():
    return [dict(key='candidate-'+name,kind='development',role='meta_validation',
                 config={'candidate':name},input_identities={'data':'b'*64},requires=[])
            for name in ('a','b')]


def test_paper_capture_is_incremental_and_survives_unavailable_historical_sources(tmp_path,monkeypatch):
    path,plan=plan_fixture(tmp_path,tasks=_two_tasks());runtime.prepare(path)
    first,second=plan['tasks'];complete(plan,first)
    stage,_,before=uploads.paper_collection(path,tmp_path/'work')
    cached=common.read(tmp_path/'work/capture-index.json')
    first_capture=cached['tasks'][first['key']]
    frozen={name:(stage/name).read_bytes() for name in first_capture['paths']}
    original=common.task_folder(plan,first);original.rename(original.with_name('unavailable-first-source'))
    complete(plan,second);copies=[];stages=[]
    copy=uploads._copy_stable;stage_file=uploads.stage_file
    def capture(source,target,**kwargs):
        copies.append(Path(source));return copy(source,target,**kwargs)
    def stage_new(base,name,*args,**kwargs):
        stages.append(Path(base));return stage_file(base,name,*args,**kwargs)
    monkeypatch.setattr(uploads,'_copy_stable',capture)
    monkeypatch.setattr(uploads,'stage_file',stage_new)
    monkeypatch.setattr(uploads,'verify_result',lambda *a,**k:pytest.fail('Paper collection must use the sealed runtime cache'))
    again,_,after=uploads.paper_collection(path,tmp_path/'work')
    assert again==stage and before['complete_tasks']==1 and after['complete_tasks']==2
    assert all((stage/name).read_bytes()==data for name,data in frozen.items())
    assert copies and all(second['key'] in source.parts for source in copies)
    assert all(second['key'] in base.parts for base in stages)
    assert common.read(tmp_path/'work/capture-index.json')['tasks'][first['key']]==first_capture


def test_paper_cycle_publishes_only_new_task_groups_and_changed_reports(tmp_path,monkeypatch):
    path,plan=plan_fixture(tmp_path,tasks=_two_tasks());runtime.prepare(path)
    first,second=plan['tasks'];complete(plan,first);calls=[];publications=[]
    class GitHub:
        def git_publish(self,files,message):
            publications.append(files);return str(len(publications))*40
    def publish(gh,round_number,files,work,*,metadata,prefix_root):
        calls.append(dict(prefix=prefix_root,names=[record['source_path'] for record in files]))
        return dict(manifest=prefix_root+'/manifest.json',status='PAPER_DATA_FILES_VERIFIED',
                    files=len(files),bytes=sum(record['source_bytes'] for record in files))
    monkeypatch.setattr(uploads,'publish_snapshot',publish)
    monkeypatch.setattr(uploads,'publish_stage_index',lambda *a,**k:None)
    work=tmp_path/'paper';gh=GitHub();one=uploads.paper_cycle(gh,path,work)
    assert len(calls)==2 and any('/per-task/'+first['key']+'/' in call['prefix'] for call in calls)
    assert uploads.paper_cycle(gh,path,work)==one and len(calls)==2 and len(publications)==1
    complete(plan,second);two=uploads.paper_cycle(gh,path,work)
    assert len(calls)==3 and '/per-task/'+second['key']+'/' in calls[-1]['prefix']
    assert all(name.startswith(second['key']+'/') for name in calls[-1]['names'])
    manifest=json.loads(next(data for name,data in publications[-1].items() if name.endswith('/manifest.json')))
    assert set(manifest['tasks'])=={first['key'],second['key']}
    assert two['complete_tasks']==2 and two['exact_original_bytes'] is True


def test_full_archive_verifies_new_task_once_even_after_network_retry(tmp_path,monkeypatch):
    path,plan=plan_fixture(tmp_path,tasks=_two_tasks());runtime.prepare(path)
    first,second=plan['tasks'];complete(plan,first);verifications=[];archived=[];indexes=[];fail=[True]
    verify=uploads.verify_result
    def verified(plan,task,**kwargs):
        verifications.append(task['key']);return verify(plan,task,**kwargs)
    def archive(gh,base,paths,work,key,identity,recipe,prefix,**kwargs):
        archived.append(key)
        if key==first['key'] and fail[0]:
            fail[0]=False;raise ConnectionError('fixture transport interruption')
        return dict(key=key,identity=identity,manifest=prefix+'/'+key+'/manifest.json',
                    parts=1,remote_full_GET_SHA256_verified=True)
    class GitHub:
        def git_publish(self,*args):return 'a'*40
    monkeypatch.setattr(uploads,'verify_result',verified)
    monkeypatch.setattr(uploads,'archive_group',archive)
    monkeypatch.setattr(uploads,'publish_stage_index',lambda gh,plan,channel,reference:indexes.append(reference))
    work=tmp_path/'full';gh=GitHub()
    with pytest.raises(ConnectionError):uploads.full_cycle(gh,path,work)
    assert verifications==[first['key']]
    assert common.read(work/'source-captures.json')['tasks'][first['key']]['identity']
    uploads.full_cycle(gh,path,work)
    assert verifications==[first['key']] and archived.count(first['key'])==2
    folder=common.task_folder(plan,first);folder.rename(folder.with_name('unavailable-first-source'))
    uploads.full_cycle(gh,path,work)
    assert verifications==[first['key']] and archived.count(first['key'])==2
    complete(plan,second);uploads.full_cycle(gh,path,work)
    assert verifications==[first['key'],second['key']] and archived.count(second['key'])==1
    assert indexes and all('groups' not in reference for reference in indexes)
    assert indexes[-1]['path'].endswith('/LATEST.json') and indexes[-1]['complete_tasks']==2


def test_archive_rejects_changed_sealed_bytes_before_remote_publication(tmp_path):
    base=tmp_path/'task';base.mkdir();source=base/'predictions.json';source.write_text('original')
    expected={source.name:dict(bytes=source.stat().st_size,sha256=common.sha(source))}
    source.write_text('changed after sealing')
    class NoNetwork:
        def ensure_release(self,*args):pytest.fail('Changed scientific evidence must never be published')
    with pytest.raises(ValueError,match='sealed completion'):
        uploads.archive_group(NoNetwork(),base,[source],tmp_path/'work','task','identity','r'*64,
                              'results/paper3/round-11',expected_files=expected)


def test_reporter_reuses_durable_completion_signature_without_task_io(tmp_path,monkeypatch):
    from types import SimpleNamespace
    path,plan=plan_fixture(tmp_path,tasks=_two_tasks());runtime.prepare(path);complete(plan,plan['tasks'][0])
    calls=[]
    def report(*args,**kwargs):calls.append(args);return SimpleNamespace(returncode=0)
    assert uploads.report_loop(path,runner=report,once=True)['status']=='PARTIAL_REPORT_GENERATED'
    task_folder=common.task_folder(plan,plan['tasks'][0]);task_folder.rename(task_folder.with_name('unavailable-source'))
    monkeypatch.setattr(uploads,'verify_result',lambda *a,**k:pytest.fail('Reporter must not inspect task directories'))
    result=uploads.report_loop(path,runner=lambda *a,**k:pytest.fail('Unchanged completions must not rerun reporting'),once=True)
    assert result['status']=='WAITING_FOR_NEW_COMPLETION' and len(calls)==1


def test_exact_report_sha_is_not_rehashed_while_its_signature_is_unchanged(tmp_path,monkeypatch):
    path,plan=plan_fixture(tmp_path);runtime.prepare(path);task=plan['tasks'][0];complete(plan,task)
    reports=Path(plan['output'])/'reports';common.atomic(reports/'report.json',{'payload':'original'})
    common.atomic(reports/'report-status.json',dict(status='COMPLETE',recipe_sha256=plan['recipe_sha256'],
        total_tasks=1,sealed_complete_tasks=1,complete_task_keys_sha256=common.digest([task['key']]),
        report_sha256=common.sha(reports/'report.json')))
    actual_sha=uploads.sha;calls=[]
    def hashed(source):calls.append(Path(source));return actual_sha(source)
    monkeypatch.setattr(uploads,'sha',hashed)
    assert uploads.report_completion(plan['output'],plan['recipe_sha256'],[task['key']])
    assert uploads.report_completion(plan['output'],plan['recipe_sha256'],[task['key']])
    assert calls==[reports/'report.json']
    common.atomic(reports/'report.json',{'payload':'changed'})
    assert not uploads.report_completion(plan['output'],plan['recipe_sha256'],[task['key']])


def test_full_campaign_collection_only_opens_the_cached_completed_task(tmp_path,monkeypatch):
    tasks=[dict(key='candidate-%05d'%number,kind='development',role='meta_validation',
                config={'candidate':number},input_identities={'data':'b'*64},requires=[])
           for number in range(10270)]
    path,plan=plan_fixture(tmp_path,tasks=tasks);runtime.prepare(path);complete(plan,tasks[-1])
    copies=[];capture=uploads._copy_stable
    def copied(source,target,**kwargs):
        copies.append(Path(source));return capture(source,target,**kwargs)
    monkeypatch.setattr(uploads,'_copy_stable',copied)
    monkeypatch.setattr(uploads,'verify_result',lambda *a,**k:pytest.fail('No task-directory scan for a 10,270-task plan'))
    _,_,metadata=uploads.paper_collection(path,tmp_path/'work')
    assert metadata['total_tasks']==10270 and metadata['complete_tasks']==1
    assert len(copies)==4  # one immutable task's three sources, and the plan
    assert all(source==path or tasks[-1]['key'] in source.parts for source in copies)
    copies.clear();uploads.paper_collection(path,tmp_path/'work')
    assert copies==[]


def test_large_stage_index_blob_fallback_keeps_head_bound_cas_merge():
    development=dict(stage='campaign',recipe_sha256='a'*64)
    formal=dict(stage='exploratory',recipe_sha256='b'*64)
    index_path=uploads.PREFIX+'/INDEX.json'
    class GitHub:
        prefix='/repos/local-fixture';branch='paper3-results'
        def __init__(self):
            self.head='head0';self.remote={};self.blobs={};self.trees={};self.commits={}
            self.read_refs=[];self.blob_reads=[];self.patches=[];self.delays=[]
        def ensure_branch(self):return self.head
        def _delay(self,attempt):self.delays.append(attempt)
        def _blob(self,data):
            identity=hashlib.sha256(data).hexdigest();self.blobs[identity]=data;return identity
        def api(self,path,data=None,method='GET'):
            if '/git/ref/heads/' in path:return {'object':{'sha':self.head}}
            if '/contents/' in path:
                self.read_refs.append(path.split('?ref=')[1])
                return dict(encoding='none',sha='index-'+self.head)
            if '/git/blobs/' in path:
                self.blob_reads.append(path.rsplit('/',1)[-1])
                return dict(encoding='base64',content=base64.b64encode(common.encoded(self.remote)).decode())
            if method=='GET' and '/git/commits/' in path:return {'tree':{'sha':'base-'+self.head}}
            if '/git/trees' in path:
                identity='tree'+str(len(self.trees));self.trees[identity]=data['tree'];return {'sha':identity}
            if method=='POST' and '/git/commits' in path:
                identity='commit'+str(len(self.commits));self.commits[identity]=data;return {'sha':identity}
            assert method=='PATCH' and '/git/refs/heads/' in path
            self.patches.append(data)
            if len(self.patches)==1:
                self.head='head1'
                self.remote=uploads.merge_stage_index({},formal,'full',{'path':'concurrent-LATEST.json'})
                raise uploads.ApiError(409)
            commit=self.commits[data['sha']];assert commit['parents']==['head1']
            entry=next(item for item in self.trees[commit['tree']] if item['path']==index_path)
            self.remote=json.loads(self.blobs[entry['sha']]);self.head=data['sha'];return {}
    gh=GitHub();uploads.publish_stage_index(gh,development,'paper',{'manifest':'candidate-index.json'})
    assert gh.read_refs==['head0','head1'] and gh.blob_reads==['index-head0','index-head1']
    assert gh.delays==[1] and all(patch['force'] is False for patch in gh.patches)
    assert gh.remote['active_stage']=='exploratory'
    assert gh.remote['stages']['exploratory']['b'*64]['channels']['full']['path']=='concurrent-LATEST.json'
    assert gh.remote['stages']['campaign']['a'*64]['channels']['paper']['manifest']=='candidate-index.json'
