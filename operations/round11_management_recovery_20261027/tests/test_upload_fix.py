from __future__ import annotations
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import urllib.error

import pytest

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import atomic_io
import upload_fix as u


def put(path,value):
    atomic_io.atomic_bytes(path,(json.dumps(value,sort_keys=True)+'\n').encode())


def read(path):return json.loads(Path(path).read_text())


@pytest.fixture
def env(tmp_path):
    runtime=tmp_path/'runtime';runtime.mkdir()
    package=tmp_path/'original-package';package.mkdir()
    science=package/'science.py';science.write_bytes(b'original scientific source\n')
    plan=tmp_path/'plan.json';put(plan,{'recipe_sha256':'a'*64,'tasks':[]})
    return dict(runtime=runtime,common=SimpleNamespace(atomic=put),path=plan,package=package,
                header=dict(recipe_sha256='a'*64,total_tasks=2))


def legacy_writer(env,channel,log,*,event=None):
    path=env['runtime']/(channel+'-upload.lock');ready=log.with_suffix('.ready');finish=log.with_suffix('.finish')
    code='''import fcntl,json,sys,time
from pathlib import Path
lock=Path(sys.argv[1]).open('a');fcntl.flock(lock,fcntl.LOCK_EX)
Path(sys.argv[2]).write_text('ready')
while not Path(sys.argv[3]).exists():time.sleep(.01)
'''
    child=subprocess.Popen([sys.executable,'-I','-S','-B','-c',code,str(path),str(ready),str(finish)])
    until=time.monotonic()+5
    while not ready.exists() and time.monotonic()<until:time.sleep(.01)
    assert ready.exists()
    return child,finish


@pytest.mark.parametrize('channel',['paper','full'])
def test_existing_live_uploader_is_observed_without_any_stop_or_reupload(env,tmp_path,monkeypatch,channel):
    cache=env['runtime']/('upload-'+channel);cache.mkdir()
    proof={'identity':'b'*64,'remote_full_GET_SHA256_verified':True}
    task=proof if channel=='full' else dict(identity='b'*64,receipt={'remote_complete_blob_GET_SHA256_verified':True})
    put(cache/('groups.json' if channel=='full' else 'published-groups.json'),
        dict(recipe_sha256='a'*64,tasks={'done':task}))
    put(cache/'status.json',dict(status='RETRYING',error_type='URLError',updated_utc='2026-10-27T03:13:52Z'))
    log=tmp_path/'legacy.log'
    log.write_text(json.dumps(dict(event='ARCHIVE_PART_VERIFIED',utc=u._utc(),name='part-00001.bin',
                                  bytes=32*1024*1024,reused=True,Authorization='never-print-secret'))+'\n')
    before=env['path'].read_bytes();source=(env['package']/'science.py').read_bytes()
    child,finish=legacy_writer(env,channel,log)
    monkeypatch.setattr(u,'_uploader',lambda env:SimpleNamespace())
    try:
        result=u.run_channel(env,channel,tmp_path/'work',once=True,launch_token='nonce',legacy_log_paths=[log])
        assert child.poll() is None
        assert result['status']=='OBSERVING_EXISTING_UPLOADER'
        assert result['current_transport_activity']=='LEGACY_UPLOADER_ACTIVE'
        assert result['observed_verified_groups']==1
        assert result['last_verified_part_utc']
        assert result['last_cycle_error']['historical_not_a_new_failure'] is True
        assert result['observer_heartbeat_not_a_new_remote_verification'] is True
        ready=read(tmp_path/'work'/(channel+'-ready.json'))
        assert ready['lock_scope']=='repair_observer' and not ready['original_upload_lock_owned']
        assert ready['launch_token']=='nonce' and ready['existing_uploader_preserved'] is True
        overlay=read(cache/'observer-status.json')
        assert 'never-print-secret' not in json.dumps(overlay)
        assert read(cache/'status.json')['updated_utc']=='2026-10-27T03:13:52Z'
        assert env['path'].read_bytes()==before and (env['package']/'science.py').read_bytes()==source
    finally:
        finish.write_text('done');child.wait(timeout=5)


def test_observer_does_not_drop_last_verified_count_on_transient_bad_json(env,tmp_path):
    cache=env['runtime']/'upload-full';cache.mkdir();path=cache/'groups.json'
    put(path,dict(recipe_sha256='a'*64,tasks={'one':dict(remote_full_GET_SHA256_verified=True)}))
    observer=u.LegacyObserver(env,'full',tmp_path)
    assert observer.observe()['observed_verified_groups']==1
    path.write_bytes(b'{temporarily unavailable')
    later=observer.observe()
    assert later['observed_verified_groups']==1 and later['upload_metadata_observation_errors']==1
    assert path.read_bytes()==b'{temporarily unavailable'


def test_old_log_error_does_not_prove_current_retry_or_activity(env,tmp_path):
    log=tmp_path/'old.log';log.write_text(json.dumps(dict(event='R11_UPLOAD_RETRY',
        utc='2026-10-27T03:13:52Z',error_type='URLError',token='do-not-log'))+'\n')
    value=u.LegacyObserver(env,'full',tmp_path,log_paths=[log]).observe(now=1793077200.)
    assert value['current_transport_activity']=='WAITING_FOR_LEGACY_EVENT_PROOF'
    assert value['last_verified_part_utc'] is None
    assert 'do-not-log' not in json.dumps(value)


def fake_uploader(env,cycle):
    return SimpleNamespace(atomic=put,full_cycle=cycle,discover_token=lambda explicit:'fake-token',
                           emit=lambda *a,**k:None)


def test_owned_missing_channel_keeps_heartbeat_during_long_cycle_then_retries(env,tmp_path,monkeypatch):
    cache=env['runtime']/'upload-full';seen=[]
    def cycle(gh,plan,work):
        until=time.monotonic()+2
        while time.monotonic()<until:
            if (work/'observer-status.json').exists():
                value=read(work/'observer-status.json')
                if value.get('status')=='RUNNING_UPLOAD_CYCLE':seen.append(value['updated_utc'])
            if len(seen)>=3:break
            time.sleep(.04)
        raise urllib.error.URLError('fake-token/private-url/secret')
    module=fake_uploader(env,cycle);monkeypatch.setattr(u,'_uploader',lambda env:module)
    original=module.atomic
    value=u.run_channel(env,'full',tmp_path/'work',once=True,heartbeat_interval=.02,
                        gh_factory=lambda *a,**k:object(),launch_token='token')
    assert len(seen)>=3 and value['status']=='RETRYING'
    assert value['last_cycle_error']['error_type']=='URLError'
    assert 'fake-token' not in json.dumps(read(cache/'status.json'))
    assert module.atomic is original
    ready=read(tmp_path/'work/full-ready.json')
    assert ready['lock_scope']=='original_upload' and ready['original_upload_lock_owned']


def test_published_group_progress_updates_incrementally_and_never_claims_whole_completion(env,tmp_path,monkeypatch):
    def cycle(gh,plan,work):
        module.atomic(work/'groups.json',dict(recipe_sha256='a'*64,tasks={'first':{}}))
        return dict(status='PARTIAL',complete_tasks=1,total_tasks=2,
                    groups={'huge':'not operational telemetry'},commit='c'*40,
                    remote_full_GET_SHA256_verified=True)
    module=fake_uploader(env,cycle);monkeypatch.setattr(u,'_uploader',lambda env:module)
    value=u.run_channel(env,'full',tmp_path/'work',once=True,gh_factory=lambda *a,**k:object())
    assert value['committed_uploaded_groups']==1
    assert value['complete_tasks']==1 and value['status']=='PARTIAL'
    assert value['remote_completion_claimed_by_heartbeat'] is False
    assert 'groups' not in value


def test_real_legacy_lock_natural_release_transfers_cached_channel_without_signals(env,tmp_path,monkeypatch):
    cache=env['runtime']/'upload-full';cache.mkdir()
    put(cache/'groups.json',dict(recipe_sha256='a'*64,tasks={'old':dict(identity='b'*64,
                                                        remote_full_GET_SHA256_verified=True)}))
    child,finish=legacy_writer(env,'full',tmp_path/'legacy.log');calls=[]
    def cycle(gh,plan,work):
        calls.append(1)
        assert 'old' in read(work/'groups.json')['tasks']
        return dict(status='COMPLETE',complete_tasks=2,total_tasks=2)
    module=fake_uploader(env,cycle);monkeypatch.setattr(u,'_uploader',lambda env:module)
    timer=threading.Timer(.15,lambda:finish.write_text('natural completion'));timer.start()
    try:
        value=u.run_channel(env,'full',tmp_path/'work',heartbeat_interval=.02,
                            gh_factory=lambda *a,**k:object())
        child.wait(timeout=5)
        assert value['status']=='COMPLETE' and calls==[1]
        assert read(tmp_path/'work/full-ready.json')['original_upload_lock_owned'] is True
    finally:
        timer.cancel();finish.write_text('finish');child.wait(timeout=5)


def test_monitor_memory_error_does_not_interrupt_upload_and_heartbeat_finishes(env,tmp_path,monkeypatch):
    writes=[]
    def atomic(path,value):
        writes.append(Path(path).name)
        if Path(path).name=='observer-status.json':raise MemoryError('telemetry only')
        put(path,value)
    env['common'].atomic=atomic
    module=fake_uploader(env,lambda *args:dict(status='PARTIAL',complete_tasks=0,total_tasks=2))
    monkeypatch.setattr(u,'_uploader',lambda env:module)
    value=u.run_channel(env,'full',tmp_path/'work',once=True,
                        gh_factory=lambda *a,**k:object(),heartbeat_interval=.01)
    assert value['status']=='PARTIAL'
    count=len(writes);time.sleep(.1);assert len(writes)==count


def test_credential_failure_is_local_retry_after_real_lock_ready(env,tmp_path,monkeypatch):
    module=fake_uploader(env,None)
    module.discover_token=lambda *args:(_ for _ in ()).throw(RuntimeError('no credentials'))
    monkeypatch.setattr(u,'_uploader',lambda env:module)
    value=u.run_channel(env,'full',tmp_path/'work',once=True)
    assert value['status']=='RETRYING'
    assert read(tmp_path/'work/full-ready.json')['status']=='READY'
    assert value['remote_completion_claimed_by_heartbeat'] is False


def test_duplicate_observer_uses_lease_without_starting_another_cycle(env,tmp_path,monkeypatch):
    work=tmp_path/'work';work.mkdir()
    monkeypatch.setattr(u,'_uploader',lambda env:SimpleNamespace())
    with (work/'full-observer.lock').open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=u.run_channel(env,'full',work,once=True)
    assert result['status']=='EXISTING_REPAIR_HELPER_REUSED'


def test_stopped_helper_never_touches_scientific_outputs_or_network(env,tmp_path,monkeypatch):
    stop=threading.Event();stop.set()
    monkeypatch.setattr(u,'_uploader',lambda env:SimpleNamespace())
    result=u.run_channel(env,'full',tmp_path/'work',stop=stop,once=True)
    assert result['status']=='STOPPED'
    assert not (tmp_path/'work/full-ready.json').exists()
