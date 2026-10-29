"""R11 export tests use original sealed contracts and deterministic archives.

The network peer is an in-memory GitHub stand-in. Scientific source verification,
packing, incremental ledgers, locks and process ownership checks are real.
"""
from __future__ import annotations
import base64
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
import urllib.parse

import pytest

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import r11_export as export

PACKAGE=Path('/workspace/paper3-round11')
if str(PACKAGE) not in sys.path:sys.path.insert(0,str(PACKAGE))
from paper3r11.platform import common, runtime, uploads


class GitHub:
    """Every fake archive upload reads and hashes its entire local part."""
    prefix='/repos/owner/repo'
    branch='paper3-results'
    def __init__(self):
        self.files={};self.parts={};self.releases={};self.calls=[];self.commits=[]
    def git_publish(self,files,message):
        self.files.update(files);self.commits.append(dict(files=dict(files),message=message))
        return hashlib.sha1(export.encoded(sorted(self.files))).hexdigest()
    def ensure_release(self,tag,title,body):
        if tag not in self.releases:self.releases[tag]=dict(id=len(self.releases)+1,tag=tag)
        return self.releases[tag]
    def upload_part(self,release,path,name,expected):
        raw=Path(path).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==expected
        key=(release['tag'],name)
        if key in self.parts:assert self.parts[key]==raw
        else:self.parts[key]=raw
        self.calls.append(dict(kind='part',release=release['tag'],name=name,sha256=expected,bytes=len(raw)))
        return dict(id=len(self.parts),url='https://example.invalid/'+release['tag']+'/'+name,
            bytes=len(raw),sha256=expected,remote_full_GET_SHA256_verified=True)
    def api(self,path,**kwargs):
        name=urllib.parse.unquote(path.split('/contents/',1)[1].split('?',1)[0])
        raw=self.files[name]
        digest=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        return dict(sha=digest,encoding='base64',content=base64.b64encode(raw).decode())


def paper_publish(gh,round_number,files,work,metadata=None,prefix_root=None):
    """Validate every staged fragment before acknowledging its exact bytes."""
    assert round_number==11
    sources=[]
    for item in files:
        raw=b''.join(Path(part['local_path']).read_bytes() for part in item['parts'])
        assert len(raw)==item['source_bytes']
        assert hashlib.sha256(raw).hexdigest()==item['source_sha256']
        sources.append(dict(source_path=item['source_path'],sha256=item['source_sha256'],bytes=len(raw)))
    target=prefix_root+'/fake-manifest.json'
    gh.git_publish({target:export.encoded(dict(metadata=metadata,files=sources))},'Fake whole-byte proof')
    return dict(remote_complete_blob_GET_SHA256_verified=True,
        collection_status=metadata['collection_status'],manifest=target,files=len(sources),
        bytes=sum(item['bytes'] for item in sources))


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    root=tmp_path/'paper3';output=root/'runs/r11_fixture';runt=output/'runtime'
    runt.mkdir(parents=True);(runt/'STOP').write_text('USER_STOPPED_SCIENCE\n')
    tasks=[dict(key='task-'+str(i),family='feature_reference',role='meta_train',requires=[]) for i in range(3)]
    plan=common.sign_plan(dict(schema_version=common.PLAN_SCHEMA,round=11,stage='development',
        root=str(root),output=str(output),package=str(PACKAGE),python=sys.executable,
        synthetic_fixture=True,code_manifest={'paper3r11/platform/common.py':export.sha(PACKAGE/'paper3r11/platform/common.py')},
        tasks=tasks))
    path=output/'plan.json';export.atomic(path,plan)
    state=dict(recipe_sha256=plan['recipe_sha256'],tasks={})
    def complete(index,binary=False):
        task=tasks[index];folder=output/task['key'];folder.mkdir(exist_ok=True)
        name='predictions.npz' if binary else 'metrics.json'
        raw=b'PK\x03\x04\xff\x00\xf0\x80exact-original-binary' if binary else b'{"accuracy": 0.75}\n'
        (folder/name).write_bytes(raw)
        artifact=dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        result=dict(status='COMPLETE',recipe_sha256=plan['recipe_sha256'],task_key=task['key'],
            task_identity=common.task_identity(plan,task),scientific_scope='SOFTWARE_TEST_ONLY',
            artifacts=[artifact],paper_artifacts=[artifact])
        export.atomic(folder/'result.json',result);common.seal_result(plan,task)
        state['tasks'][task['key']]=dict(phase='COMPLETE',task_identity=result['task_identity'],
            result_sha256=export.sha(folder/'result.json'),completion_sha256=export.sha(folder/'completion.json'))
        export.atomic(runt/'tasks.json',state)
    complete(0);complete(1)
    (output/'reports').mkdir();(output/'reports/report.json').write_text('{"status":"PARTIAL","completed":2}\n')
    args=SimpleNamespace(root=str(root),work=str(tmp_path/'exports'),plan=str(path),r11_plan=str(path),credentials=None)
    env=export.load_environment(args);gh=GitHub()
    monkeypatch.setattr(env['u'],'publish_snapshot',paper_publish)
    return SimpleNamespace(args=args,env=env,gh=gh,plan=plan,state=state,tasks=tasks,complete=complete,
        output=output,runtime=runt,stop=(runt/'STOP').read_bytes())


def group(env,key,row):
    return dict(key=key,identity=row['completion_sha256'],manifest='old-verified/'+key,
        commit='old-commit',source_files=3,parts=1,remote_full_GET_SHA256_verified=True)


def saved_paper(env,rows):
    return dict(recipe_sha256=env['plan']['recipe_sha256'],tasks={key:dict(identity=uploads._capture_identity(row),
        receipt=dict(remote_complete_blob_GET_SHA256_verified=True,collection_status='COMPLETE',
            manifest='old-paper/'+key)) for key,row in rows.items()},
        reports=dict(identity='old-reports',receipt=dict(remote_complete_blob_GET_SHA256_verified=True,
            manifest='old-paper-reports',collection_status='PARTIAL')))


def test_real_archive_reuses_native_group_and_publishes_actual_completed_set(fixture):
    f=fixture;env=f.env;work=f.runtime/'upload-full';work.mkdir()
    rows=export.completion_rows(env)
    prior=group(env,'task-0',rows['task-0'])
    export.atomic(work/'groups.json',dict(recipe_sha256=f.plan['recipe_sha256'],tasks={'task-0':prior}))
    scientific={p:p.read_bytes() for p in f.output.glob('task-*/*') if p.is_file()}
    first=export.full_once(f.args,env=env,client=f.gh)
    assert first['status']=='UPLOADING' and first['uploaded_tasks']==2 and first['task_key']=='task-1'
    assert len(f.gh.parts)==1
    archive=next(iter(f.gh.parts.values()))
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as handle:
        assert handle.extractfile('metrics.json').read()==scientific[f.output/'task-1/metrics.json']
        assert set(handle.getnames())=={'metrics.json','result.json','completion.json'}
    ledger=export.read(work/'groups.json')
    assert ledger['tasks']['task-0']==prior
    second=export.full_once(f.args,env=env,client=f.gh)
    assert second['complete'] and second['available_completed_tasks']==2
    assert not second['all_planned_task_receipts_complete'] and not second['scientific_completion_claimed']
    pointer=json.loads(f.gh.files[export.PREFIX+'/COMPLETED_FULL_LATEST.json'])
    assert pointer['status']=='AVAILABLE_COMPLETED_OUTPUT_UPLOADED'
    assert pointer['observed_completed_tasks']==pointer['uploaded_completed_tasks']==2
    assert pointer['planned_tasks']==3 and pointer['scientific_completion_claimed'] is False
    manifest=json.loads(f.gh.files[pointer['manifest']])
    assert set(manifest['tasks'])=={'task-0','task-1'}
    assert manifest['reports']['key']=='plan-and-reports'
    assert manifest['remote_full_GET_SHA256_verified'] is True
    calls=len(f.gh.calls);commits=len(f.gh.commits)
    third=export.full_once(f.args,env=env,client=f.gh)
    assert third['complete'] and len(f.gh.calls)==calls and len(f.gh.commits)==commits
    assert all(path.read_bytes()==raw for path,raw in scientific.items())
    assert (f.runtime/'STOP').read_bytes()==f.stop


def test_full_uses_same_lock_without_competing_upload(fixture):
    f=fixture
    with export.lock(f.runtime/'full-upload.lock'):
        result=export.full_once(f.args,env=f.env,client=f.gh)
    assert result['status']=='WAITING_EXISTING_FULL_UPLOADER'
    assert not result['complete'] and not f.gh.calls and not f.gh.commits


def test_final_archive_includes_only_existing_scientific_job_log_snapshot(fixture):
    f=fixture;logs=f.runtime/'logs';logs.mkdir()
    (logs/'slot-0-123.out').write_text('TASK COMPLETE\n')
    (logs/'slot-0-123.err').write_text('historical worker stderr\n')
    nested=logs/'worker';nested.mkdir();(nested/'kernel.log').write_text('scientific worker trace\n')
    (logs/'manager.log').write_text('never export management logs\n')
    (f.runtime/'full-upload.log').write_text('never export uploader logs\n')
    (logs/'external.out').symlink_to(f.runtime/'full-upload.log')
    (logs/'other').symlink_to(f.runtime,target_is_directory=True)
    rows=export.completion_rows(f.env);work=f.runtime/'upload-full';work.mkdir()
    export.atomic(work/'groups.json',dict(recipe_sha256=f.plan['recipe_sha256'],
        tasks={key:group(f.env,key,row) for key,row in rows.items()}))
    result=export.full_once(f.args,env=f.env,client=f.gh)
    assert result['complete']
    with tarfile.open(fileobj=io.BytesIO(next(iter(f.gh.parts.values()))),mode='r:gz') as handle:
        names=set(handle.getnames())
        assert {'runtime/logs/slot-0-123.out','runtime/logs/slot-0-123.err',
            'runtime/logs/worker/kernel.log'}<=names
        assert not any(name.endswith(('manager.log','full-upload.log','external.out')) for name in names)
        assert handle.extractfile('runtime/logs/slot-0-123.out').read()==b'TASK COMPLETE\n'
    manifest=json.loads(f.gh.files[result['pointer']['manifest']])
    assert 'not sealed scientific task evidence' in manifest['ancillary_snapshot_scope']
    assert manifest['scientific_completion_claimed'] is False


def test_progress_manifests_use_small_delta_and_final_index_is_complete(fixture):
    f=fixture;rows=export.completion_rows(f.env)
    groups={key:group(f.env,key,row) for key,row in rows.items()}
    first=export._publish_summary(f.gh,f.env,'full',rows,{'task-0':groups['task-0']},{},
        complete=False,delta_group='task-0')
    second=export._publish_summary(f.gh,f.env,'full',rows,groups,{},
        complete=False,delta_group='task-1')
    delta=json.loads(f.gh.files[second['manifest']])
    assert delta['task_index_mode']=='DELTA' and delta['previous_manifest']==first['manifest']
    assert set(delta['tasks'])=={'task-1'} and delta['uploaded_completed_tasks']==2
    assert delta['manifest_includes_all_uploaded_task_records'] is False
    final=export._publish_summary(f.gh,f.env,'full',rows,groups,{},complete=True)
    whole=json.loads(f.gh.files[final['manifest']])
    assert whole['task_index_mode']=='FULL' and whole['previous_manifest'] is None
    assert set(whole['tasks'])==set(rows)
    assert final['available_completed_output_complete'] is True


def test_frozen_completion_metadata_refreshes_only_on_state_change(fixture,monkeypatch):
    f=fixture;calls=[];original=f.env['rt'].completed_task_rows
    def observed(plan):calls.append(1);return original(plan)
    monkeypatch.setattr(f.env['rt'],'completed_task_rows',observed)
    assert len(export.completion_rows(f.env))==2
    assert len(export.completion_rows(f.env))==2 and len(calls)==1
    f.complete(2)
    assert len(export.completion_rows(f.env))==3 and len(calls)==2
    assert len(export.completion_rows(f.env))==3 and len(calls)==2


def test_changed_scientific_artifact_is_never_acknowledged(fixture):
    f=fixture;(f.output/'task-0/metrics.json').write_text('{"accuracy":999}\n')
    with pytest.raises(ValueError,match='changed after sealing'):
        export.full_once(f.args,env=f.env,client=f.gh)
    assert not f.gh.parts and not f.gh.commits
    assert not (f.runtime/'upload-full/completed-export-status.json').exists()
    assert (f.runtime/'STOP').read_bytes()==f.stop


def test_complete_cached_paper_can_be_ready_while_original_daemon_is_alive(fixture):
    f=fixture;work=f.runtime/'upload-paper';work.mkdir()
    rows=export.completion_rows(f.env);saved=saved_paper(f.env,rows)
    export.atomic(work/'published-groups.json',saved)
    with export.lock(f.runtime/'paper-upload.lock'):
        result=export.paper_once(f.args,env=f.env,client=f.gh)
    assert result['paper_ready'] and result['complete'] and result['uploaded_tasks']==2
    assert not result['all_planned_task_receipts_complete']
    assert export.read(work/'published-groups.json')==saved
    pointer=json.loads(f.gh.files[export.PREFIX+'/COMPLETED_PAPER_LATEST.json'])
    assert pointer['remote_complete_blob_GET_SHA256_verified'] is True
    assert pointer['remote_full_GET_SHA256_verified'] is False
    assert pointer['scientific_completion_claimed'] is False


def test_stale_native_report_proof_is_refreshed_from_current_exact_source(fixture):
    f=fixture;work=f.runtime/'upload-paper';work.mkdir()
    rows=export.completion_rows(f.env);saved=saved_paper(f.env,rows)
    export.atomic(work/'published-groups.json',saved)
    report=f.output/'reports/report.json';fresh=b'{"status":"PARTIAL","completed":2,"fresh":true}\n'
    report.write_bytes(fresh)
    result=export.paper_once(f.args,env=f.env,client=f.gh)
    state=export.read(f.env['work']/'paper-report-snapshot.json')
    proof=state['reports']['receipt'];assert proof['manifest']!='old-paper-reports'
    manifest=json.loads(f.gh.files[proof['manifest']])
    row=next(item for item in manifest['files'] if item['source_path']=='reports/report.json')
    assert row['sha256']==hashlib.sha256(fresh).hexdigest() and row['bytes']==len(fresh)
    assert result['paper_ready'] and export.read(work/'published-groups.json')==saved
    commits=len(f.gh.commits)
    assert export.paper_once(f.args,env=f.env,client=f.gh)['paper_ready']
    assert len(f.gh.commits)==commits


def test_own_paper_report_cache_refreshes_while_original_daemon_holds_lock(fixture):
    f=fixture;work=f.runtime/'upload-paper';work.mkdir()
    export.atomic(work/'published-groups.json',saved_paper(f.env,export.completion_rows(f.env)))
    export.paper_once(f.args,env=f.env,client=f.gh)
    previous=export.read(f.env['work']/'paper-report-snapshot.json')
    fresh=b'{"status":"PARTIAL","completed":2,"revision":12345}\n'
    (f.output/'reports/report.json').write_bytes(fresh)
    legacy_before=(work/'published-groups.json').read_bytes()
    with export.lock(f.runtime/'paper-upload.lock'):
        result=export.paper_once(f.args,env=f.env,client=f.gh)
    current=export.read(f.env['work']/'paper-report-snapshot.json')
    assert result['paper_ready'] and current['source_signatures']!=previous['source_signatures']
    assert current['reports']['identity']!=previous['reports']['identity']
    manifest=json.loads(f.gh.files[current['reports']['receipt']['manifest']])
    row=next(item for item in manifest['files'] if item['source_path']=='reports/report.json')
    assert row['sha256']==hashlib.sha256(fresh).hexdigest()
    assert (work/'published-groups.json').read_bytes()==legacy_before


def test_missing_paper_waits_for_original_daemon(fixture):
    f=fixture
    with export.lock(f.runtime/'paper-upload.lock'):
        result=export.paper_once(f.args,env=f.env,client=f.gh)
    assert result['status']=='WAITING_EXISTING_PAPER_UPLOADER'
    assert result['paper_ready'] is False and not f.gh.commits


def test_missing_binary_paper_is_losslessly_encoded_and_scoped(fixture):
    f=fixture;f.complete(0,binary=True)
    # Make the existing second task and report receipts reusable. Only task0
    # requires an exact new paper capture in this one metadata process.
    rows=export.completion_rows(f.env);saved=saved_paper(f.env,rows)
    del saved['tasks']['task-0']
    work=f.runtime/'upload-paper';work.mkdir();export.atomic(work/'published-groups.json',saved)
    original=(f.output/'task-0/predictions.npz').read_bytes()
    result=export.paper_once(f.args,env=f.env,client=f.gh)
    assert result['status']=='UPLOADING' and result['task_key']=='task-0'
    ledger=export.read(work/'published-groups.json');receipt=ledger['tasks']['task-0']['receipt']
    assert receipt['transport_representations_changed'] is True
    mapping=receipt['binary_representations'][0]
    assert mapping['original_source_path']=='task-0/predictions.npz'
    assert mapping['decoded_sha256']==hashlib.sha256(original).hexdigest()
    assert mapping['strict_base64_validated'] is True
    assert mapping['representation_source_path'].endswith('.b64.txt')
    assert (f.output/'task-0/predictions.npz').read_bytes()==original
    assert export.paper_once(f.args,env=f.env,client=f.gh)['paper_ready']


def test_full_refuses_foreign_ledger_without_resetting_history(fixture):
    f=fixture;path=f.runtime/'upload-full/groups.json'
    original=dict(recipe_sha256='1'*64,tasks={'foreign':{'manifest':'prior'}});export.atomic(path,original)
    with pytest.raises(ValueError,match='LEDGER_RECIPE_DIFFERS'):
        export.full_once(f.args,env=f.env,client=f.gh)
    assert export.read(path)==original and not f.gh.calls


def test_canonical_plan_is_required_for_complete_report_archive(fixture):
    f=fixture;(f.output/'plan.json').write_text('{}')
    with pytest.raises(ValueError,match='PLAN_DIFFERS'):
        export._report_paths(f.env)


def test_exact_full_uploader_identity_only(fixture):
    f=fixture;path=f.runtime/'upload-full/process.json'
    argv=[sys.executable,'-I','-S','-B','-u',str(PACKAGE/'paper3r11/platform/entry.py'),
        'upload','--plan',str(f.env['path']),'--channel','full']
    record=dict(pid=2468,uid=os.getuid(),start_ticks='123',argv=argv)
    export.atomic(path,dict(pid=2468,channel='full'))
    process=lambda pid:record if pid==2468 else None
    result=export.legacy_uploader_candidates(f.args,'full',process=process,env=f.env)
    assert len(result)==1 and result[0]['kind']=='uploader' and result[0]['role']=='full'
    assert result[0]['binding_plan']==str(f.env['path']) and result[0]['start_ticks']=='123'
    for bad in (
        argv[:-1]+['paper'],
        argv[:6]+['worker']+argv[7:],
        argv+['--child'],
        argv[:5]+[str(PACKAGE/'paper3r11/platform/science.py')]+argv[6:],
        argv+['--channel','full'],
    ):
        record['argv']=bad
        assert not export.legacy_uploader_candidates(f.args,'full',process=process,env=f.env)
    record['argv']=argv
    export.atomic(path,dict(pid=2468,channel='full',start_ticks='reused-pid'))
    assert not export.legacy_uploader_candidates(f.args,'full',process=process,env=f.env)


def test_metadata_reader_rejects_malformed_and_oversized_instead_of_resetting(tmp_path):
    path=tmp_path/'metadata.json';path.write_text('malformed')
    with pytest.raises(json.JSONDecodeError):export.read(path,{})
    path.write_text('{"a":"value"}')
    with pytest.raises(ValueError,match='SIZE_EXCEEDED'):export.read(path,{},max_bytes=4)
    assert export.read(tmp_path/'missing.json',{})=={}
