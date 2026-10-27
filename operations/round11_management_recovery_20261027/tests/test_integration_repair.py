"""CPU-only integration with the original R11 contract and actual archive bytes.

Scientific fixture completions are explicitly software-only. GitHub transport
is in memory, Slurm observation is a local runner, and no work is submitted.
"""
from __future__ import annotations

import base64
import fcntl
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path('/workspace/paper3-round11')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ORIGINAL))


@pytest.fixture
def real_contract(tmp_path):
    import paper3r11
    from paper3r11.platform import common, runtime, scheduler, uploads
    import environment
    # Other compatibility tests may already have imported the same unchanged
    # production package from its repository checkout. Bind that exact module
    # root rather than deliberately crossing the process's package guard.
    package = Path(paper3r11.__file__).resolve().parent.parent
    saved = [(module, key, getattr(module, key))
             for name, module in list(sys.modules.items())
             if name.startswith('paper3r11.') and module is not None
             for key in ('atomic', 'atomic_bytes', 'read', 'load_plan', '_state',
                         '_read', '_capture_paper_task', 'paper_collection',
                         'paper_cycle', 'publish_snapshot', 'publish_stage_index',
                         'channel_loop', 'discover_token') if hasattr(module, key)]
    old_load, old_query, old_atomic = (scheduler.FourJobScheduler._load,
                                     scheduler.FourJobScheduler._query, scheduler._atomic)
    installation = getattr(uploads, '_paper_binary_sidecar_installation', None)
    callback = getattr(uploads, '_paper_sidecar_ready_callback', None)
    root = tmp_path/'school'; root.mkdir()
    output = root/'runs/r11_campaign'; rt = output/'runtime'; rt.mkdir(parents=True)
    tasks = [dict(key='reference-'+name, kind='feature_reference', family='feature_reference',
                  role='meta_train', config={}, requires=[])
             for name in ('complete', 'active', 'missing-real-source')]
    plan = common.sign_plan(dict(schema_version=common.PLAN_SCHEMA, round=11,
        stage='campaign', synthetic_fixture=True, root=str(root), output=str(output),
        package=str(package), python=sys.executable, tasks=tasks,
        comparison_family=[], secondary_ablation_family=[], reference_strategy='fixture-fixed',
        resources=dict(jobs=4,gpus_per_job=1,cpus_per_job=16,memory_per_job='64G'),
        code_manifest={'paper3r11/platform/common.py':common.sha(package/'paper3r11/platform/common.py')}))
    path = output/'plan.json'; common.atomic(path, plan)
    launch = runtime.context(path)['launch']; common.atomic(rt/'launch.json', launch)
    common.atomic(rt/'SUBMIT_AUTHORIZATION.json', dict(explicit_submit=True,
        plan_sha256=common.sha(path), recipe_sha256=plan['recipe_sha256']))
    task = tasks[0]; folder = common.task_folder(plan,task);folder.mkdir()
    binary = bytes(range(256))*47
    (folder/'operator.npz').write_bytes(binary)
    (folder/'participants.csv').write_bytes(b'participant_id,ba\r\n"software-fixture",0.5\r\n')
    files = [dict(path=name,bytes=(folder/name).stat().st_size,sha256=common.sha(folder/name))
             for name in ('operator.npz','participants.csv')]
    common.atomic(folder/'result.json', dict(status='COMPLETE', scientific_scope='SOFTWARE_TEST_ONLY',
        recipe_sha256=plan['recipe_sha256'],task_key=task['key'],
        task_identity=common.task_identity(plan,task),artifacts=files,
        paper_artifacts=[item['path'] for item in files]))
    common.seal_result(plan,task)
    rows = {task['key']:dict(phase='COMPLETE',task_identity=common.task_identity(plan,task),
        result_sha256=common.sha(folder/'result.json'),completion_sha256=common.sha(folder/'completion.json')),
        tasks[1]['key']:dict(phase='RUNNING',attempt=2,slot=1,job_id='101',
            attempt_token='b'*64,consecutive_failures=0),
        tasks[2]['key']:dict(phase='BLOCKED_CONFIGURATION',attempt=1,
            consecutive_failures=1,current_error=dict(error_type='FileNotFoundError',reason='Actual pinned fixture source missing'),
            failure_history=[dict(attempt=1,error_type='FileNotFoundError')])}
    common.atomic(rt/'tasks.json',dict(recipe_sha256=plan['recipe_sha256'],tasks=rows))
    marker = environment.legacy.scheduler_identity(rt,plan['recipe_sha256'])
    slots = {str(slot):dict(attempts=[dict(slot=slot,sequence=1,marker=marker,
        name=f'{marker}-s{slot}-a1',status='ACKNOWLEDGED',job_id=str(100+slot),intent_at=1.)],
        next_submit_at=0.,rejections=0,worker_failures=0) for slot in range(4)}
    common.atomic(rt/'scheduler-state.json',dict(version=1,marker=marker,created_at=1.,slots=slots))
    env = environment.load_env(package,path,expected_recipe=None)
    before = {p: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    before[path] = path.read_bytes(); before[rt/'tasks.json']=(rt/'tasks.json').read_bytes()
    def runner(command,**kwargs):
        if command[0]=='squeue':
            text='\n'.join(f'{100+s}|{marker}|RUNNING|{marker}-s{s}-a1|fixture-gpu|fixture-gpu' for s in range(4))
            return SimpleNamespace(returncode=0,stdout=text,stderr='')
        if command[0]=='sacct':return SimpleNamespace(returncode=0,stdout='',stderr='')
        raise AssertionError('Integration must not submit or cancel jobs: '+command[0])
    try:
        yield SimpleNamespace(env=env,plan=plan,path=path,work=tmp_path/'management',
                              common=common,runtime=runtime,uploads=uploads,runner=runner,
                              before=before,binary=binary,folder=folder)
    finally:
        for module,key,value in saved:setattr(module,key,value)
        scheduler.FourJobScheduler._load=old_load;scheduler.FourJobScheduler._query=old_query
        scheduler._atomic=old_atomic
        for key,value in [('_paper_binary_sidecar_installation',installation),
                          ('_paper_sidecar_ready_callback',callback)]:
            if value is None:uploads.__dict__.pop(key,None)
            else:setattr(uploads,key,value)


class MemoryGitHub:
    """Exact archive upload and git-file readback, entirely in memory."""
    prefix='repos/local/software-fixture'
    prefix_url='https://api.github.com/repos/local/software-fixture'
    branch='paper3-results'
    def __init__(self):
        self.commits={};self.assets={};self.publications=[]
    def ensure_branch(self):return '0'*40
    def ensure_release(self,tag,*_):return dict(id=tag,tag_name=tag)
    def upload_part(self,release,path,name,sha256):
        data=Path(path).read_bytes();assert hashlib.sha256(data).hexdigest()==sha256
        self.assets[(release['id'],name)]=data
        return dict(id=len(self.assets),size=len(data),sha256=sha256,
                    remote_full_GET_SHA256_verified=True)
    def git_publish(self,files,message):
        ref='%040x'%(len(self.commits)+1);self.commits[ref]=dict(files)
        self.publications.append((message,dict(files)));return ref
    def api(self,path,*_,**__):
        from urllib.parse import unquote
        if '/contents/' in path:
            name=unquote(path.split('/contents/',1)[1].split('?ref=',1)[0])
            ref=path.split('?ref=',1)[1];data=self.commits[ref][name]
            blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            return dict(sha=blob,encoding='base64',content=base64.b64encode(data).decode())
        raise AssertionError('Unexpected local GitHub endpoint')


def test_original_contract_controller_report_and_archive_roles_preserve_live_science(real_contract,monkeypatch):
    import controller, report_fix, upload_fix
    from paper3r11 import report
    from paper3r11.platform.vendor import lossless_publish
    f=real_contract;u=f.uploads;c=f.common
    observed=controller.controller_loop(f.path,f.work,package=f.env['package'],submit=True,
        expected_recipe=None,once=True,runner=f.runner,launch_token='fixture-controller')
    assert observed['task_status']['complete_tasks']==1
    assert observed['task_status']['blocked_tasks']==1
    assert observed['first_cycle_submit_allowed'] is False
    assert len(observed['jobs'])==4
    def actual_cpu_report(command,**kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']==''
        value=report.generate_report(f.path)
        assert value['coverage']['sealed_complete_tasks']==1
        return SimpleNamespace(returncode=0)
    generated=report_fix.run_report(f.env,f.work,once=True,runner=actual_cpu_report,
                                    launch_token='fixture-report')
    assert generated['report_complete_tasks']==1
    assert generated['status']=='PARTIAL_REPORT_GENERATED'
    gh=MemoryGitHub();snapshots=[]
    def verified_text_snapshot(client,number,files,work,*,metadata,prefix_root):
        for record in files:
            data=b''.join(lossless_publish._part_bytes(part) for part in record['parts'])
            assert len(data)==record['source_bytes']
            assert hashlib.sha256(data).hexdigest()==record['source_sha256']
            data.decode('utf-8')
            if record['source_path'].endswith('.b64.txt'):
                assert base64.b64decode(data,validate=True)==f.binary
        snapshots.append(list(files))
        return dict(manifest=prefix_root+'/manifest.json',files=len(files),
                    bytes=sum(file['source_bytes'] for file in files),
                    status='PAPER_DATA_PARTIAL_FILES_VERIFIED',exact_original_bytes=True)
    monkeypatch.setattr(u,'publish_snapshot',verified_text_snapshot)
    monkeypatch.setattr(u,'publish_stage_index',lambda *args,**kwargs:None)
    monkeypatch.setattr(u,'discover_token',lambda *_:'software-fixture-secret')
    factory=lambda *args,**kwargs:gh
    paper=upload_fix.run_channel(f.env,'paper',f.work,once=True,gh_factory=factory,
                                 launch_token='fixture-paper')
    full=upload_fix.run_channel(f.env,'full',f.work,once=True,gh_factory=factory,
                                launch_token='fixture-full')
    assert paper['status'] not in {'RETRYING','FAILED'}
    assert full['status']=='PARTIAL'
    assert snapshots and gh.assets
    archived=[]
    for data in gh.assets.values():
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
            if 'operator.npz' in archive.getnames():archived.append(archive.extractfile('operator.npz').read())
    assert f.binary in archived
    monitored=report_fix.run_monitor(f.env,f.work,once=True,client_factory=lambda:gh,
                                     launch_token='fixture-monitor')
    assert monitored['status']=='PUBLISHED'
    for path,data in f.before.items():assert path.read_bytes()==data
    assert c.verify_result(f.plan,f.plan['tasks'][0])
    for role in ('controller','report','paper','full','monitor'):
        ready=c.read(f.work/(role+'-ready.json'))
        assert ready['status']=='READY' and ready['launch_token']=='fixture-'+role
    state=f.runtime._state(f.plan)
    assert state['tasks']['reference-active']['attempt']==2
    assert state['tasks']['reference-missing-real-source']['failure_history']


def test_live_full_transfer_is_observed_without_touching_its_lock_cache_or_science(real_contract):
    import upload_fix
    f=real_contract;c=f.common;cache=Path(f.env['runtime'])/'upload-full';cache.mkdir()
    error=dict(status='RETRYING',error_type='URLError',updated_utc='2026-10-27T03:13:52Z')
    c.atomic(cache/'status.json',error)
    c.atomic(cache/'groups.json',dict(recipe_sha256=f.plan['recipe_sha256'],tasks={
        'already-uploaded':dict(identity='a'*64,remote_full_GET_SHA256_verified=True)}))
    log=Path(f.env['runtime'])/'full-upload.log'
    log.write_text(json.dumps(dict(event='ARCHIVE_PART_VERIFIED',utc=c.utc(),
                                  name='part-00001.bin',bytes=32*1024*1024,reused=True))+'\n')
    with (Path(f.env['runtime'])/'full-upload.lock').open('a') as ongoing:
        fcntl.flock(ongoing,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def forbidden(*args,**kwargs):
            raise AssertionError('Existing full transfer must not create a new network client')
        result=upload_fix.run_channel(f.env,'full',f.work,once=True,gh_factory=forbidden,
                                      launch_token='fixture-observer')
        assert result['status']=='OBSERVING_EXISTING_UPLOADER'
        assert result['original_upload_lock_owned'] is False
        assert result['existing_uploader_preserved'] is True
        assert result['observed_verified_groups']==1
        assert result['last_verified_part_utc'] is not None
        assert result['observer_heartbeat_not_a_new_remote_verification'] is True
        with (Path(f.env['runtime'])/'full-upload.lock').open('a') as competing:
            with pytest.raises(BlockingIOError):fcntl.flock(competing,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert c.read(cache/'status.json')==error
    assert c.read(cache/'groups.json')['tasks']['already-uploaded']['identity']=='a'*64
    for path,data in f.before.items():assert path.read_bytes()==data
