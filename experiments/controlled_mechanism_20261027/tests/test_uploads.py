from __future__ import annotations
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from paper3m12 import uploads as u
from paper3m12.common import atomic_json, digest, read_json, sha_file


class FakeGit:
    def __init__(self):
        self.blobs={};self.trees={'t0':{'older-round/evidence.txt':'oldblob'}}
        self.commits={'c0':{'tree':{'sha':'t0'},'parents':[]}}
        self.head='c0';self.counter=0;self.post_blobs=0;self.get_blobs=0
        self.conflict=False;self.corrupt=False;self.fail_publish=None
        self.published=[]

    def request(self,path,body=None,method=None,allow404=False):
        method=method or ('GET' if body is None else 'POST')
        leaf=path.split('/git/',1)[-1]
        if leaf=='blobs' and method=='POST':
            assert body['encoding']=='base64'
            data=base64.b64decode(body['content'],validate=True)
            identity=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            self.blobs[identity]=data;self.post_blobs+=1
            return {'sha':identity}
        if leaf.startswith('blobs/'):
            self.get_blobs+=1
            data=self.blobs[leaf.split('/')[-1]]+(b'corrupt' if self.corrupt else b'')
            return {'encoding':'base64','content':base64.encodebytes(data).decode()}
        if leaf.startswith('ref/heads/'):
            return {'object':{'sha':self.head}}
        if leaf.startswith('commits/'):
            return self.commits[leaf.split('/')[-1]]
        if leaf=='trees':
            tree=dict(self.trees[body['base_tree']]);tree.update({entry['path']:entry['sha'] for entry in body['tree']})
            key='t'+str(len(self.trees));self.trees[key]=tree
            return {'sha':key}
        if leaf=='commits':
            key='c'+str(len(self.commits));self.commits[key]={'tree':{'sha':body['tree']},'parents':body['parents']}
            self.commits[key]['message']=body['message']
            return {'sha':key}
        if leaf.startswith('refs/heads/') and method=='PATCH':
            value=self.commits[body['sha']]
            assert body['force'] is False
            if self.fail_publish and self.fail_publish(value['message']):
                raise u.TransportError(503)
            if self.conflict:
                self.conflict=False
                tree=dict(self.trees[self.commits[self.head]['tree']['sha']]);tree['concurrent-round/keep.txt']='externalblob'
                self.trees['external-tree']=tree
                self.commits['external']={'tree':{'sha':'external-tree'},'parents':[self.head]}
                self.head='external'
                raise u.TransportError(422)
            if value['parents']!=[self.head]:raise u.TransportError(422)
            self.head=body['sha'];self.published.append(self.head)
            return {'object':{'sha':self.head}}
        raise AssertionError((path,method))

    def files(self):
        return self.trees[self.commits[self.head]['tree']['sha']]


@pytest.fixture
def client(tmp_path):
    fake=FakeGit();client=u.GitHubClient('DO_NOT_LOG_THIS_SECRET',request=fake.request,cache_path=tmp_path/'cache.jsonl')
    return client,fake


def sealed_task(plan,task,values=None):
    folder=Path(plan['output'])/'tasks'/task['key'];folder.mkdir(parents=True,exist_ok=True)
    values=values or {'metrics.csv':b'method,BA,balanced_NLL\r\ntop_q,0.6,0.5\r\n',
                       'predictions.bin':b'\x00\xff\xfe\x80binary checkpoint\n'*20,
                       'participant_summary.json':b'{"scientific":"summary"}\n'}
    files=[]
    for name,data in values.items():
        path=folder/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        files.append(dict(path=path.relative_to(plan['output']).as_posix(),bytes=len(data),sha256=sha_file(path)))
    result=dict(status='COMPLETE',task_key=task['key'],task_identity=digest(task),
                recipe_sha256=plan['recipe_sha256'],files=files)
    for item in files:
        if item['path'].endswith('/metrics.csv'):result['metrics_csv']=item
    atomic_json(folder/'result.json',result)
    return result


@pytest.fixture
def study(tmp_path):
    tasks=[dict(key='one',kind='paired_block',role='pilot',participant='p1',seed=20261018),
           dict(key='two',kind='paired_block',role='pilot',participant='p2',seed=20261018)]
    plan=dict(root=str(tmp_path),output=str(tmp_path/'output'),runtime=str(tmp_path/'runtime'),
              tasks=tasks,recipe_sha256='f'*64,config={},github_credentials='/private/secret-path.json')
    sealed_task(plan,tasks[0]);return plan


def restore_manifest(manifest,fake):
    chunks=[]
    for index,item in enumerate(manifest['parts']):
        data=fake.blobs[item['git_blob_sha']]
        assert hashlib.sha256(data).hexdigest()==item['sha256']
        if index and manifest['representation'].startswith('csv-'):data=data[item['header_bytes']:]
        chunks.append(data)
    restored=b''.join(chunks)
    assert len(restored)==manifest['original_bytes']
    assert hashlib.sha256(restored).hexdigest()==manifest['original_sha256']
    return restored


def test_binary_blob_api_roundtrip_and_verified_cache_survives_restart(client,tmp_path,capsys):
    active,fake=client;data=bytes(range(256))*100
    identity=active.blob(data)
    assert fake.blobs[identity]==data and fake.get_blobs==1
    assert active.blob(data)==identity and fake.post_blobs==1
    resumed=u.GitHubClient('DO_NOT_LOG_THIS_SECRET',request=fake.request,cache_path=tmp_path/'cache.jsonl')
    assert resumed.blob(data)==identity and fake.post_blobs==1
    assert 'DO_NOT_LOG' not in capsys.readouterr().out
    assert 'DO_NOT_LOG' not in (tmp_path/'cache.jsonl').read_text()


def test_remote_full_blob_verification_refuses_corruption(client):
    active,fake=client;fake.corrupt=True
    with pytest.raises(u.TransportError):active.blob(b'\xff\x00\x80')
    assert not active.cache


def test_csv_parts_preserve_all_records_quoted_newlines_and_exact_bytes(client,tmp_path):
    active,fake=client
    original=b'key,text\r\n1,"first\r\nsecond"\r\n2,"escaped ""quote"""\r\n3,last'
    path=tmp_path/'metrics.csv';path.write_bytes(original)
    entries,manifest=u.upload_file(active,'test/csv',path,csv_chunk_bytes=26)
    assert len(manifest['parts'])>=2
    assert restore_manifest(manifest,fake)==original
    for item in manifest['parts']:
        assert fake.blobs[item['git_blob_sha']].startswith(b'key,text\r\n')
        assert item['path'].endswith('.csv')
    assert all(name in entries for name in [item['path'] for item in manifest['parts']])


def test_arbitrary_binary_split_and_zero_byte_file_restore_exactly(client,tmp_path):
    active,fake=client;path=tmp_path/'predictions.npz';path.write_bytes(bytes(range(256))*19)
    _,manifest=u.upload_file(active,'test/binary',path,binary_chunk_bytes=37)
    assert restore_manifest(manifest,fake)==path.read_bytes()
    assert all(part['bytes']<=37 for part in manifest['parts'])
    empty=tmp_path/'empty.bin';empty.write_bytes(b'')
    _,manifest=u.upload_file(active,'test/empty',empty,binary_chunk_bytes=37)
    assert restore_manifest(manifest,fake)==b'' and len(manifest['parts'])==1


def test_compare_and_swap_retry_preserves_other_round_and_concurrent_commit(client):
    active,fake=client;fake.conflict=True
    blob=active.blob(b'new')
    active.publish({'round12/new.txt':blob},'new evidence')
    assert fake.files()['older-round/evidence.txt']=='oldblob'
    assert fake.files()['concurrent-round/keep.txt']=='externalblob'
    assert fake.files()['round12/new.txt']==blob


def test_full_archive_contains_all_declared_bytes_and_no_private_extras(study,tmp_path):
    task=study['tasks'][0];result=read_json(Path(study['output'])/'tasks/one/result.json')
    private=Path(study['output'])/'tasks/one/private';private.mkdir();(private/'github.json').write_text('secret')
    one=u.make_task_archive(study,task,result,tmp_path/'one.tar.gz')
    two=u.make_task_archive(study,task,result,tmp_path/'two.tar.gz')
    assert one.read_bytes()==two.read_bytes()
    with tarfile.open(one,'r:gz') as archive:
        names=archive.getnames()
        assert sorted(names)==sorted([item['path'] for item in result['files']]+['tasks/one/result.json'])
        for name in names:assert archive.extractfile(name).read()==(Path(study['output'])/name).read_bytes()
        assert all('private' not in name for name in names)


def test_upload_channels_are_independent_and_partial_is_not_science_complete(study,client):
    active,fake=client
    before={str(path):sha_file(path) for path in Path(study['output']).rglob('*') if path.is_file()}
    paper=u.run_once(study,'paper',client=active,publish_progress=False)
    assert paper['status']=='PARTIAL' and paper['complete_tasks']==1 and paper['total_tasks']==2
    assert paper['scientific_completion_claimed'] is False
    paper_manifest=json.loads(fake.blobs[fake.files()[paper['tasks']['one']['manifest_path']]])
    names=[item['original_name'] for item in paper_manifest['content']['files']]
    assert 'metrics.csv' in names and 'predictions.bin' not in names
    full=u.run_once(study,'full',client=active,publish_progress=False)
    assert full['status']=='PARTIAL' and full['complete_tasks']==1
    full_manifest=json.loads(fake.blobs[fake.files()[full['tasks']['one']['manifest_path']]])
    recovered=restore_manifest(full_manifest['content'],fake)
    with tarfile.open(fileobj=io.BytesIO(recovered),mode='r:gz') as archive:
        assert archive.extractfile('tasks/one/predictions.bin').read()==(Path(study['output'])/'tasks/one/predictions.bin').read_bytes()
    assert before=={str(path):sha_file(path) for path in Path(study['output']).rglob('*') if path.is_file()}
    public=json.loads(fake.blobs[fake.files()[u.PREFIX+'/studies/'+'f'*16+'/paper/plan.json']])
    assert 'github_credentials' not in public


def test_failure_resume_never_records_uncommitted_task_or_stops_training(study,client,capsys):
    active,fake=client;sealed_task(study,study['tasks'][1])
    fake.fail_publish=lambda message:message.endswith('task two')
    result=u.run_once(study,'paper',client=active,publish_progress=False)
    assert result['status']=='RETRYING' and set(result['tasks'])=={'one'}
    assert result['training_controlled'] is False
    assert len(list(u.reports.completed_results(study)))==2
    assert not any(path.name in ('STOP','stop.json') for path in Path(study['output']).rglob('*'))
    assert 'DO_NOT_LOG_THIS_SECRET' not in capsys.readouterr().out
    posts=fake.post_blobs;fake.fail_publish=None
    resumed=u.run_once(study,'paper',client=active,publish_progress=False)
    assert resumed['status']=='COMPLETE' and set(resumed['tasks'])=={'one','two'}
    assert fake.post_blobs-posts<5  # Already remote-verified files are reused.
    assert resumed['scientific_completion_claimed'] is False


def test_damaged_local_archive_rebuilt_on_resumable_failure(study,client):
    active,fake=client;fake.fail_publish=lambda message:True
    first=u.run_once(study,'full',client=active,publish_progress=False)
    assert first['status']=='RETRYING'
    archive=next((Path(study['runtime'])/'upload-work/full/archives').glob('*.tar.gz'))
    archive.write_bytes(b'damaged cached archive')
    fake.fail_publish=None
    assert u.run_once(study,'full',client=active,publish_progress=False)['status']=='PARTIAL'
    with tarfile.open(archive,'r:gz') as value:assert 'tasks/one/predictions.bin' in value.getnames()


def test_progress_refreshed_without_new_tasks_and_upload_state_is_compact(study,client,monkeypatch):
    active,fake=client;calls=[]
    import paper3m12.runtime as runtime
    monkeypatch.setattr(runtime,'status',lambda plan:calls.append(1) or dict(status='RUNNING',queue_observation=len(calls)))
    factory=lambda branch:active
    u._publish_progress(study,{'status':'PARTIAL','tasks':{'x':'large'},'task_manifests':{'x':'large'}},{},client_factory=factory)
    u._publish_progress(study,{'status':'PARTIAL','tasks':{'x':'large'}},{},client_factory=factory)
    result=json.loads(fake.blobs[fake.files()['monitoring/paper3-round12/status.json']])
    assert result['queue_observation']==2 and len(calls)==2
    assert 'tasks' not in result['uploads']['paper']
    assert 'task_manifests' not in result['uploads']['paper']
    assert result['scientific_completion_claimed_by_upload'] is False


def test_token_discovery_uses_existing_private_config_without_emitting_it(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(Path,'home',lambda:tmp_path)
    for name in ('PAPER3_GITHUB_TOKEN','GITHUB_TOKEN','GH_TOKEN','PAPER3_GITHUB_CREDENTIALS'):monkeypatch.delenv(name,raising=False)
    path=tmp_path/'.config/paper3/github.json';path.parent.mkdir(parents=True);atomic_json(path,{'token':'SENSITIVE_PRIVATE_TOKEN'})
    assert u.discover_token()=='SENSITIVE_PRIVATE_TOKEN'
    assert capsys.readouterr().out==''


def test_report_failure_is_independent_retry_not_a_scientific_receipt(study,monkeypatch):
    monkeypatch.setattr(u.reports,'build_report',lambda plan:(_ for _ in ()).throw(MemoryError('details must not escape')))
    result=u.report_once(study)
    assert result['status']=='RETRYING' and result['error_type']=='MemoryError'
    assert result['training_controlled'] is False
    assert read_json(Path(study['output'])/'tasks/one/result.json')['status']=='COMPLETE'


def test_public_plan_preserves_frozen_provenance_without_any_credential_registry(study):
    source_hash='a'*64;dictionary_hash='b'*64;input_hash='c'*64;code_hash='d'*64
    study.update(input_identity=input_hash, code_manifest={'paper3m12/worker.py':code_hash},
        phase_task_counts={'paired_block':2}, scientific_scope='development and old-cohort exploration',
        independent_confirmation={'status':'NOT_AVAILABLE','planned_evaluable_participants':90},
        github_token='TOP_LEVEL_SECRET', input_registry={'token':'OLD_REGISTRY_SECRET'},
        native_inputs={'input_identity':input_hash, 'schema_version':'frozen-inputs',
            'source_by_seed':{'20261018':{'checkpoint':{'path':'runs/source/model.pt','sha256':source_hash,'bytes':150},
                                       'identity':{'source_checkpoint_sha256':source_hash},
                                       'github_token':'NESTED_SOURCE_SECRET'}},
            'dictionary_by_seed':{'20261018':{'checkpoint':{'path':'runs/dictionary/b.pt','sha256':dictionary_hash,'bytes':80},
                                           'native_identity':'e'*64}},
            'internal_data_identity':{'manifest_sha256':'f'*64},
            'people':{'pilot':['P1'],'reference':['P2']},
            'reference_rule':'64 fixed disjoint training events',
            'registry':{'token':'NATIVE_REGISTRY_SECRET','external_manifests':{'password':'NO'}},
            'credentials':{'token':'OTHER_SECRET'}})
    study['config']['credentials']={'token':'CONFIG_SECRET'}
    public=u._public_plan(study)
    assert public['input_identity']==input_hash
    assert public['code_manifest']['paper3m12/worker.py']==code_hash
    assert public['phase_task_counts']['paired_block']==2
    assert public['independent_confirmation']['status']=='NOT_AVAILABLE'
    native=public['native_inputs']
    assert native['source_by_seed']['20261018']['checkpoint']['sha256']==source_hash
    assert native['dictionary_by_seed']['20261018']['checkpoint']['sha256']==dictionary_hash
    assert native['people']['pilot']==['P1'] and native['reference_rule'].startswith('64 fixed')
    raw=json.dumps(public)
    assert 'SECRET' not in raw and 'credentials' not in raw
    assert 'registry' not in native and 'input_registry' not in public
