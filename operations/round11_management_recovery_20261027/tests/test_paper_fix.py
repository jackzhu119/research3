"""CPU/local-only paper sidecar checks against the unchanged production R11 API."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import fcntl
import os
import urllib.parse

import numpy as np
import pytest

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,'/workspace/paper3-round11')
import paper_fix
from paper3r11 import algorithm
from paper3r11.platform import common,runtime,uploads
from paper3r11.platform.vendor import lossless_publish


@pytest.fixture(autouse=True)
def restore_module():
    names=('_capture_paper_task','paper_collection','paper_cycle','publish_snapshot','publish_stage_index','load_plan',
           'channel_loop','discover_token')
    original={name:getattr(uploads,name) for name in names}
    marker=getattr(uploads,'_paper_binary_sidecar_installation',None)
    ready=getattr(uploads,'_paper_sidecar_ready_callback',None)
    yield
    for name,value in original.items():setattr(uploads,name,value)
    if marker is None:
        uploads.__dict__.pop('_paper_binary_sidecar_installation',None)
    else:uploads._paper_binary_sidecar_installation=marker
    if ready is None:uploads.__dict__.pop('_paper_sidecar_ready_callback',None)
    else:uploads._paper_sidecar_ready_callback=ready


def fixture_plan(tmp_path,keys=('reference-a','reference-b','operator')):
    package=Path('/workspace/paper3-round11');root=tmp_path/'scientific-root';root.mkdir()
    tasks=[dict(key=key,role='meta_train',family='feature_reference' if key.startswith('reference') else 'operator_train',
                config={},requires=[]) for key in keys]
    plan=common.sign_plan(dict(schema_version=common.PLAN_SCHEMA,round=11,stage='campaign',synthetic_fixture=True,
        root=str(root),output=str(root/'runs/r11-unchanged-science'),package=str(package),tasks=tasks,
        code_manifest={'paper3r11/platform/common.py':common.sha(package/'paper3r11/platform/common.py')}))
    path=root/'plan.json';common.atomic(path,plan)
    return path,plan


def complete(plan,key,contents):
    task=next(task for task in plan['tasks'] if task['key']==key)
    folder=Path(plan['output'])/key;folder.mkdir(parents=True,exist_ok=True)
    for name,data in contents.items():
        target=folder/name;target.parent.mkdir(parents=True,exist_ok=True)
        if isinstance(data,bytes):target.write_bytes(data)
        else:common.atomic(target,data)
    artifacts=[dict(path=name,bytes=(folder/name).stat().st_size,sha256=common.sha(folder/name)) for name in contents]
    common.atomic(folder/'result.json',dict(status='COMPLETE',scientific_scope='SOFTWARE_TEST_ONLY',
        recipe_sha256=plan['recipe_sha256'],task_key=key,task_identity=common.task_identity(plan,task),
        artifacts=artifacts,paper_artifacts=list(contents)))
    common.seal_result(plan,task)
    row=dict(phase='COMPLETE',task_identity=common.task_identity(plan,task),
             result_sha256=common.sha(folder/'result.json'),completion_sha256=common.sha(folder/'completion.json'))
    state_path=Path(plan['output'])/'runtime/tasks.json'
    state=common.read(state_path,dict(recipe_sha256=plan['recipe_sha256'],tasks={}))
    state['tasks'][key]=row;common.atomic(state_path,state)
    return row


def native_operator_bytes(tmp_path):
    support=np.array([[-1.,0.],[-1.,0.],[1.,0.],[1.,0.]])
    future=np.array([[0.,-1.],[0.,-1.],[0.,1.],[0.,1.]])
    pair=dict(support_features=support,support_logits=np.zeros((4,2)),support_y=np.array([0,0,1,1]),
        future_features=future,future_logits=np.zeros((4,2)),future_y=np.array([0,0,1,1]),
        participant_id='synthetic',session_id='synthetic-session',training_role='meta_train',
        support_ids=['s0','s1','s2','s3'],future_ids=['q0','q1','q2','q3'])
    operator=algorithm.train_operator([pair],algorithm.Config(feature_rank=2,include_intercept=False,ridge=.03),
        reference=dict(mean=np.zeros(2),whitener=np.eye(2),eigenvectors=np.eye(2),role='meta_train'))
    path=tmp_path/'production-format-operator.npz';algorithm.save_operator(operator,path)
    algorithm.load_operator(path,expected_operator_identity=operator.operator_identity)
    return path.read_bytes(),operator.operator_identity


def staged_bytes(record):
    data=b''.join(lossless_publish._part_bytes(part) for part in record['parts'])
    assert len(data)==record['source_bytes'] and hashlib.sha256(data).hexdigest()==record['source_sha256']
    return data


def test_original_native_operator_failure_is_utf8_not_same_name_cache_collision(tmp_path):
    _,plan=fixture_plan(tmp_path);work=tmp_path/'paper';stage=tmp_path/'captures'
    refs=[]
    for number,key in enumerate(('reference-a','reference-b')):
        row=complete(plan,key,{'reference.json':{'seed':20261018+number,'schema_version':'paper3-r11-feature-reference-v1'}})
        refs.append(uploads._capture_paper_task(plan,key,row,stage,work))
    assert not ({file['cache_manifest'] for file in refs[0]['files']} & {file['cache_manifest'] for file in refs[1]['files']})
    binary,_=native_operator_bytes(tmp_path)
    row=complete(plan,'operator',{'operator.npz':binary,'operator.json':{'format':'production-operator'}})
    with pytest.raises(lossless_publish.ValidationError,match='^source_utf8_invalid$'):
        uploads._capture_paper_task(plan,'operator',row,stage,work)


def test_native_npz_and_text_are_losslessly_represented_with_provenance(tmp_path):
    path,plan=fixture_plan(tmp_path);binary,operator_id=native_operator_bytes(tmp_path)
    text=b'participant_id,metric\r\n"S01",0.5\r\n'
    row=complete(plan,'operator',{'operator.npz':binary,'participants.csv':text,'operator.json':{'identity':operator_id}})
    protected=[path]+list((Path(plan['output'])/'operator').iterdir())
    protected+= [Path(uploads.__file__),Path(lossless_publish.__file__),Path(algorithm.__file__)]
    before={str(source):common.sha(source) for source in protected if source.is_file()}
    originals={name:getattr(uploads,name) for name in ('full_cycle','monitor_cycle','report_loop','stage_file','publish_snapshot')}
    installed=paper_fix.install_patch(uploads)
    assert installed==paper_fix.install_patch(uploads)
    assert all(getattr(uploads,name) is function for name,function in originals.items())
    record=uploads._capture_paper_task(plan,'operator',row,tmp_path/'captures',tmp_path/'paper')
    assert record['exact_original_bytes_scope']==paper_fix.SCOPE and len(record['binary_representations'])==1
    files={record['source_path']:record for record in record['files']}
    assert 'operator/operator.npz' not in files
    representation='operator/binary-base64/operator.npz.b64.txt'
    encoded=staged_bytes(files[representation]);decoded=base64.b64decode(encoded,validate=True)
    assert decoded==binary
    mapping=record['binary_representations'][0]
    assert mapping['original_source_path']=='operator/operator.npz'
    assert mapping['representation_source_path']==representation
    assert mapping['original_bytes']==len(binary) and mapping['original_sha256']==hashlib.sha256(binary).hexdigest()
    assert mapping['encoded_bytes']==len(encoded) and mapping['encoded_sha256']==hashlib.sha256(encoded).hexdigest()
    assert mapping['strict_base64_validated'] and mapping['original_completion_sha256']==row['completion_sha256']
    manifest=json.loads(staged_bytes(files['operator/binary-base64/REPRESENTATIONS.json']))
    assert manifest['representations']==record['binary_representations'] and 'strict' in manifest['reconstruction']
    assert staged_bytes(files['operator/participants.csv'])==text
    restored=tmp_path/'restored-operator.npz';restored.write_bytes(decoded)
    algorithm.load_operator(restored,expected_operator_identity=operator_id)
    assert before=={str(source):common.sha(source) for source in protected if source.is_file()}


def test_multi_block_base64_strict_decode_preserves_all_binary_values(tmp_path):
    binary=bytes(range(256))*12307
    _,plan=fixture_plan(tmp_path,keys=('operator',))
    row=complete(plan,'operator',{'operator.npz':binary})
    paper_fix.install_patch(uploads)
    record=uploads._capture_paper_task(plan,'operator',row,tmp_path/'capture',tmp_path/'paper')
    encoded=staged_bytes(next(file for file in record['files'] if file['source_path'].endswith('.b64.txt')))
    assert base64.b64decode(encoded,validate=True)==binary
    bad=tmp_path/'bad.b64';bad.write_bytes(base64.b64encode(b'first')+base64.b64encode(b'after-padding'))
    with pytest.raises(paper_fix.PaperRepairValidationError):paper_fix.strict_decode_proof(bad,len(binary),hashlib.sha256(binary).hexdigest())


def test_legal_old_capture_and_published_receipt_are_reused_with_all_scopes(tmp_path,monkeypatch):
    path,plan=fixture_plan(tmp_path);work=tmp_path/'paper'
    complete(plan,'reference-a',{'reference.json':{'seed':20261018}})
    _,_,_=uploads.paper_collection(path,work)
    state=uploads._paper_capture_state(work,plan);old=state['tasks']['reference-a']
    old_receipt=dict(manifest='existing-text-manifest.json',files=3,bytes=100,commit='a'*40)
    common.atomic(work/'published-groups.json',dict(recipe_sha256=plan['recipe_sha256'],
        tasks={'reference-a':dict(identity=old['identity'],receipt=old_receipt)}))
    binary,_=native_operator_bytes(tmp_path)
    complete(plan,'operator',{'operator.npz':binary,'operator.json':{'format':'production'}})
    calls=[];remote=[]
    def local_snapshot(gh,number,files,work,*,metadata,prefix_root):
        assert all(staged_bytes(file).decode('utf-8') is not None for file in files)
        calls.append(dict(prefix=prefix_root,files=files,metadata=metadata))
        return dict(manifest=prefix_root+'/manifest.json',files=len(files),bytes=sum(file['source_bytes'] for file in files),
                    status='PAPER_DATA_PARTIAL_FILES_VERIFIED',exact_original_bytes=True)
    class LocalGitHub:
        prefix='/repos/local-fixture'
        def __init__(self):self.commits={}
        def git_publish(self,files,message):
            remote.append(files);commit='%040x'%len(remote);self.commits[commit]=files;return commit
        def api(self,path):
            name=urllib.parse.unquote(path.split('/contents/',1)[1].split('?ref=',1)[0])
            ref=path.split('?ref=',1)[1]
            contents=self.commits[ref][name]
            return dict(sha=hashlib.sha1(b'blob '+str(len(contents)).encode()+b'\0'+contents).hexdigest())
    monkeypatch.setattr(uploads,'publish_snapshot',local_snapshot)
    monkeypatch.setattr(uploads,'publish_stage_index',lambda *a,**k:None)
    paper_fix.install_patch(uploads)
    receipt=uploads.paper_cycle(LocalGitHub(),path,work)
    assert all('/per-task/reference-a/' not in call['prefix'] for call in calls)
    assert uploads._paper_capture_state(work,plan)['tasks']['reference-a']==old
    assert receipt['exact_original_bytes_scope']==paper_fix.SCOPE and receipt['binary_representations']
    binary_group=next(call for call in calls if '/per-task/operator/' in call['prefix'])
    assert binary_group['metadata']['declared_paper_artifacts_exact_unsanitized_bytes'] is False
    assert binary_group['metadata']['original_scientific_files_modified'] is False
    manifests=[json.loads(data) for batch in remote for name,data in batch.items() if name.endswith('/manifest.json')]
    assert manifests[-1]['exact_original_bytes_scope']==paper_fix.SCOPE
    assert manifests[-1]['metadata']['declared_paper_artifacts_exact_unsanitized_bytes'] is False
    pointers=[json.loads(data) for batch in remote for name,data in batch.items() if name.endswith('PAPER_DATA_LATEST.json')]
    assert all(pointer['exact_original_bytes_scope']==paper_fix.SCOPE for pointer in pointers)


def test_scope_pointer_readback_mismatch_cannot_reuse_old_path_verified_flag(tmp_path,monkeypatch):
    def snapshot(*args,**kwargs):
        return dict(manifest='old-verified-manifest.json',published_index_paths_verified=True,
                    status='PAPER_DATA_PARTIAL_FILES_VERIFIED')
    class WrongPath:
        prefix='/repos/local-fixture'
        def git_publish(self,*args):return 'a'*40
        def api(self,path):return dict(sha='0'*40)
    monkeypatch.setattr(uploads,'publish_snapshot',snapshot)
    with pytest.raises(paper_fix.PaperRepairValidationError,match='^published_index_path_mismatch$'):
        paper_fix._publish_scoped(uploads,WrongPath(),[],tmp_path,{},'results/fixed-prefix',[])


def test_bad_raw_binary_capture_is_repaired_without_changing_originals(tmp_path):
    path,plan=fixture_plan(tmp_path,keys=('operator',));work=tmp_path/'paper';binary,_=native_operator_bytes(tmp_path)
    row=complete(plan,'operator',{'operator.npz':binary})
    identity=uploads._capture_identity(row);stage=work/'captured'/plan['recipe_sha256']
    stale=dict(identity=identity,recipe_sha256=plan['recipe_sha256'],task_key='operator',files=[{'source_path':'operator/operator.npz'}],
        paths=[],coverage=dict(task_key='operator',captured_paper_artifacts=[],has_declared_paper_artifacts=True))
    common.atomic(work/'capture-index.json',dict(recipe_sha256=plan['recipe_sha256'],tasks={'operator':stale},reports={}))
    common.atomic(stage/'per-task/operator'/identity/'capture.json',stale)
    original_sha=common.sha(Path(plan['output'])/'operator/operator.npz')
    paper_fix.install_patch(uploads);_,_,metadata=uploads.paper_collection(path,work)
    capture=uploads._paper_capture_state(work,plan)['tasks']['operator']
    assert capture['binary_representations'] and not any(file['source_path'].endswith('/operator.npz') for file in capture['files'])
    assert metadata['declared_paper_artifacts_exact_unsanitized_bytes'] is False
    assert common.sha(Path(plan['output'])/'operator/operator.npz')==original_sha


def test_corrupt_binary_representation_is_rejected_by_original_sha_verifier(tmp_path):
    _,plan=fixture_plan(tmp_path,keys=('operator',));binary,_=native_operator_bytes(tmp_path)
    row=complete(plan,'operator',{'operator.npz':binary});paper_fix.install_patch(uploads)
    record=uploads._capture_paper_task(plan,'operator',row,tmp_path/'capture',tmp_path/'paper')
    encoded=next(file for file in record['files'] if file['source_path'].endswith('.b64.txt'))
    part=Path(encoded['parts'][0]['local_path']);data=part.read_bytes();part.write_bytes(b'!'+data[1:])
    with pytest.raises(lossless_publish.ValidationError,match='^staged_part_SHA256_changed$'):
        staged_bytes(encoded)


def test_diagnostic_fixed_code_is_allowlisted_and_never_echoes_exception_body(tmp_path,monkeypatch,capsys):
    secret='NOT_A_REAL_TOKEN_do_not_echo';paper_fix.install_patch(uploads)
    monkeypatch.setattr(uploads,'load_plan',lambda *a,**k:(_ for _ in ()).throw(ValueError(secret)))
    with pytest.raises(ValueError):uploads.paper_cycle(object(),tmp_path/'plan.json',tmp_path/'paper')
    assert secret not in capsys.readouterr().out
    diagnostic=json.loads((tmp_path/'paper/paper-fix-diagnostic.json').read_text())
    assert diagnostic['fixed_code']=='paper_error_code_not_allowlisted'
    assert secret not in json.dumps(diagnostic)
    assert paper_fix.diagnostic_code(lossless_publish.ValidationError('source_utf8_invalid'))=='source_utf8_invalid'


def test_ready_callback_follows_actual_original_paper_lock_and_precedes_network(tmp_path,monkeypatch):
    path,plan=fixture_plan(tmp_path,keys=('operator',));ready=[]
    runtime_path=uploads.runtime_dir(plan)
    def callback(payload):
        assert (runtime_path/'upload-paper/process.json').is_file()
        process=common.read(runtime_path/'upload-paper/process.json')
        assert process['pid']==os.getpid()
        with (runtime_path/'paper-upload.lock').open('a') as probe:
            with pytest.raises(BlockingIOError):fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ready.append(payload)
    def offline(*args,**kwargs):
        assert ready and ready[0]['lock_acquired'] is True
        raise ConnectionError('synthetic transport failure after ready')
    monkeypatch.setattr(uploads,'discover_token',lambda *_:'synthetic-no-network-token')
    paper_fix.install_patch(uploads,ready_callback=callback)
    result=uploads.channel_loop(path,'paper',gh_factory=offline,once=True)
    assert result['status']=='RETRYING' and len(ready)==1
    assert ready[0]['pid']==os.getpid() and ready[0]['recipe_sha256']==plan['recipe_sha256']
    assert ready[0]['role']=='paper' and ready[0]['patch_id']==paper_fix.REPAIR_ID
    assert ready[0]['remote_paper_upload_complete'] is False


def test_busy_paper_lock_never_emits_false_ready(tmp_path):
    path,plan=fixture_plan(tmp_path,keys=('operator',));ready=[]
    runtime_path=uploads.runtime_dir(plan);runtime_path.mkdir(parents=True)
    paper_fix.install_patch(uploads,ready_callback=ready.append)
    with (runtime_path/'paper-upload.lock').open('a') as existing:
        fcntl.flock(existing,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=uploads.channel_loop(path,'paper',once=True)
    assert result['status']=='EXISTING_PROCESS_REUSED' and ready==[]
