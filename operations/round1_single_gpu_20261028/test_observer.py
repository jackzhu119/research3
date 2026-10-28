import base64
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

HERE=Path(__file__).resolve().parent


def module(name):
    spec=importlib.util.spec_from_file_location('_observer_test_'+name,HERE/(name+'.py'))
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


@pytest.fixture
def fixture(tmp_path):
    obs=module('observer');root=tmp_path/'root';home=tmp_path/'home';work=home/'single'
    runtime=root/obs.OPERATIONAL
    runtime.mkdir(parents=True);work.mkdir(parents=True)
    obs.atomic(runtime/'progress.json',dict(status='RUNNING_OR_QUEUED',phase='MAINLINE_FINALIZATION',
        updated_utc='2026-10-28T14:52:04Z',tasks=[dict(key='done',kind='development',status='COMPLETE'),
            dict(key='strongest-control',kind='strongest_control',status='RUNNING',slot=0,job_id='123',
                 expected_receipt='runs/training_campaign_v1/strongest_control.json')],
        jobs=[dict(job_id='122',slot=0,cpus=12,memory='43G')],mainline_complete=False,all_original_complete=False))
    obs.atomic(work/'controller-status.json',dict(status='RUNNING',phase='MAINLINE_FINALIZATION',
        scheduler=dict(jobs=[dict(job_id='123',slot=0,cpus=4,memory='64G',partition='hpc_gpu')]),
        cumulative_updates=999))
    obs.atomic(work/'allocation-state.json',dict(status='RUNNING'))
    obs.atomic(runtime/'queue.json',dict(status='RUNNING'))
    token='a'*32;key='strongest-control'
    obs.atomic(runtime/'claims.json',{key:dict(slot=0,job_id='123',token=token)})
    obs.atomic(runtime/'worker-status/slot-0.json',dict(slot=0,job_id='123',phase='RUNNING',
        action_key=key,action_kind='strongest_control',claim_token=token,child_pid=44,
        action_progress=[dict(file='progress.json',values=dict(outer_updates_completed=22,
            observed_optimizer_calls_total=99,private='SHOULD_NOT_PUBLISH'))]))
    log=runtime/'actions'/hashlib.sha256(key.encode()).hexdigest()/'attempts'/token/'kernel.log'
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps(dict(event='R1_FILE_DIGEST_CACHE_PROGRESS',hits=100,bytes_reused=222))+'\n'+
        json.dumps(dict(event='R1_PROGRESS',outer_updates_completed=22,token='ghp_'+('s'*36)))+'\n'+
        'Traceback (most recent call last):\nMemoryError: arbitrary ghp_'+('x'*36)+'\n')
    def queue(command,**kwargs):
        assert command[0]=='squeue' and kwargs['timeout']==8
        return subprocess.CompletedProcess(command,0,'123|paper3r1s-abc-a1|RUNNING|hpc_gpu|1:00|gpu2|gpu2\n'+
            '456|paper3r11p-something|RUNNING|hpc_gpu|2:00|gpu1|gpu1\n','')
    return obs,root,home,work,runtime,queue


def test_collect_readonly_actual_resources_and_fresh_queue(fixture):
    obs,root,home,work,runtime,queue=fixture
    before={p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result=obs.collect(root,home,work,runner=queue)
    assert result['parallel_plan_complete_tasks']==1 and result['parallel_plan_total_tasks']==2
    assert result['resources_target']['jobs']==result['resources_target']['gpus_total']==1
    assert result['resources_target']['cpus_per_job']==4
    assert result['resources_target']['memory_total']=='64G'
    assert result['jobs'][0]['job_id']=='123' and result['jobs'][0]['live_state']=='RUNNING'
    assert len(result['live_queue']['jobs'])==1
    assert result['coordinator']['updated_utc']=='2026-10-28T14:52:04Z'
    assert result['single_gpu_controller']['read_error'] is None
    assert before=={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_claim_bound_events_and_numerical_counts_do_not_leak(fixture):
    obs,root,home,work,runtime,queue=fixture
    result=obs.collect(root,home,work,runner=queue)
    worker=result['single_gpu_worker'];assert worker['claim_bound']
    assert worker['actual_numeric_counters'][0]['values']['outer_updates_completed']==22
    assert worker['current_action_log']['observed_error_types']==['MemoryError']
    assert worker['current_action_log']['raw_log_text_published'] is False
    raw=json.dumps(result)
    assert 'ghp_' not in raw and 'SHOULD_NOT_PUBLISH' not in raw and 'arbitrary' not in raw
    assert 'hits' not in raw and 'bytes_reused' not in raw


def test_unbound_worker_never_reads_attempt_log(fixture):
    obs,root,home,work,runtime,queue=fixture
    obs.atomic(runtime/'claims.json',{})
    worker=obs.collect(root,home,work,runner=queue)['single_gpu_worker']
    assert not worker['claim_bound'] and 'current_action_log' not in worker


def test_missing_progress_not_claimed_complete(fixture):
    obs,root,home,work,runtime,queue=fixture
    (runtime/'progress.json').unlink()
    snapshot=obs.collect(root,home,work,runner=queue)
    assert snapshot['parallel_plan_total_tasks'] is None and snapshot['parallel_plan_complete_tasks'] is None
    assert snapshot['all_original_complete'] is False
    assert snapshot['coordinator']['read_error']=='FileNotFoundError'


def test_unavailable_queue_does_not_label_jobs_gone(fixture):
    obs,root,home,work,runtime,_=fixture
    snapshot=obs.collect(root,home,work,runner=lambda *a,**k:subprocess.CompletedProcess(a,1,'','denied'))
    assert snapshot['live_queue']['status']=='UNAVAILABLE'
    assert 'live_state' not in snapshot['jobs'][0]


def test_once_fresh_children_and_retry_is_nonfatal(fixture):
    obs,root,home,work,runtime,_=fixture;calls=[]
    def runner(cmd,timeout,*cap):
        calls.append((cmd,timeout))
        if '--collect-cycle' in cmd:return dict(updated_utc='2026-10-28T15:00:00Z',parallel_plan_complete_tasks=3,
            parallel_plan_total_tasks=4,coordinator=dict(phase='internal'))
        assert '--publish-cycle' in cmd
        return dict(status='RETRYING',reason_code='NETWORK_ERROR',http_status=None)
    status=obs.once(root,home,work,runner=runner)
    assert status['status']=='RETRYING' and status['training_controlled'] is False
    assert [x[1] for x in calls]==[30,52]
    assert all(cmd[1:5]==['-I','-S','-B','-u'] for cmd,_ in calls)
    assert (work/'observer-snapshot.json').exists() and (work/'observer-status.json').exists()


def test_own_helper_timeout_does_not_signal_other_process(tmp_path):
    obs=module('observer')
    other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            obs._bounded([sys.executable,'-c','import time;time.sleep(30)'],0.03)
        assert other.poll() is None
    finally:
        other.terminate();other.wait(timeout=3)


def test_existing_monitor_lock_prevents_duplicate_once(fixture,capsys):
    obs,root,home,work,runtime,_=fixture
    with (runtime/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert obs.main(['--root',str(root),'--home',str(home),'--work',str(work),'--once'])==0
    assert 'ALREADY_RUNNING' in capsys.readouterr().out
    assert not (work/'observer-snapshot.json').exists()


def test_new_status_same_schema_keeps_extensions_unstarted(fixture):
    obs,root,home,work,runtime,queue=fixture
    snapshot=obs.collect(root,home,work,runner=queue)
    assert snapshot['schema_version']=='paper3-round1-parallel-monitor-v1'
    assert snapshot['mainline']['observed_actions']==2
    assert all(x['status']=='NOT_STARTED' for x in snapshot['extension_arms'].values())


def test_process_record_rechecks_pid_and_argv():
    obs=module('observer');record=obs.process_record(os.getpid())
    assert record and obs.process_matches(record)
    assert not obs.process_matches(dict(record,start_ticks='0'))
    assert not obs.process_matches(dict(record,argv=['fake']))


def test_legacy_discovery_requires_exact_root_and_runtime(fixture,tmp_path):
    obs,root,home,work,runtime,_=fixture;proc=tmp_path/'proc';proc.mkdir()
    def fake(pid,args):
        p=proc/str(pid);p.mkdir();(p/'cmdline').write_bytes(b'\0'.join(x.encode() for x in args)+b'\0')
        # state(field3) then18 fields then startticks(field22)
        (p/'stat').write_text(str(pid)+' (python) S '+'0 '*18+'1234 0 0')
    fake(11,['python','/old/monitor.py','--root',str(root),'--operational',str(runtime)])
    fake(12,['python','/old/monitor.py','--root',str(root),'--operational','/other'])
    fake(13,['python','/old/train.py','--root',str(root),'--operational',str(runtime)])
    found=obs.legacy_monitor_candidates(root,home,work,proc=proc)
    assert len(found)==1 and found[0]['pid']==11 and found[0]['start_ticks']=='1234'


def test_actual_isolated_collect_cli_and_bytes_unchanged(fixture,tmp_path):
    obs,root,home,work,runtime,_=fixture
    bindir=tmp_path/'bin';bindir.mkdir();sq=bindir/'squeue';sq.write_text('#!/bin/sh\nexit 0\n');sq.chmod(0o700)
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    cmd=obs._command('--collect-cycle',root,home,work)
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=10,
        env=dict(os.environ,PATH=str(bindir)+':'+os.environ['PATH']))
    assert result.returncode==0,result.stderr
    snapshot=json.loads(result.stdout)
    assert snapshot['resources_target']['cpus_per_job']==4
    assert snapshot['live_queue']==dict(status='OBSERVED',observed_utc=snapshot['live_queue']['observed_utc'],jobs=[])
    assert before=={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_fixed_progress_transport_pin_get_verification(tmp_path):
    transport=module('upload_transport');report=b'{"round":1,"status":"RUNNING"}'
    calls=[];saved={};commit='a'*40;blob='b'*40
    def request(endpoint,token,body,method,timeout):
        calls.append((endpoint,body,method));assert token=='TEST_TOKEN'
        if endpoint=='git/ref/heads/paper3-r1-progress':return dict(object=dict(sha=commit))
        if endpoint.endswith('?ref=paper3-r1-progress'):
            raise transport.DeliveryError('HTTP_ERROR',404)
        if method=='PUT':
            saved.update(body);return dict(commit=dict(sha=commit),content=dict(sha=blob))
        if endpoint.endswith('?ref='+commit):return dict(encoding='base64',content=base64.b64encode(report).decode())
        raise AssertionError(endpoint)
    result=transport.publish(report,tmp_path,_request=request,_token=lambda *_:'TEST_TOKEN',emit=lambda _:None)
    assert result['status']=='PROGRESS_UPLOADED_VERIFIED'
    assert result['complete_GET_SHA256_verified']
    assert saved['branch']=='paper3-r1-progress'
    assert all('diagnostics/' not in x[0] for x in calls)
    assert any('monitoring/paper3-round1/status.json?ref='+commit==x[0].removeprefix('contents/') for x in calls)


def test_transport_never_publishes_credential_in_snapshot(tmp_path):
    transport=module('upload_transport')
    result=transport.publish(b'{"secret":"TEST_TOKEN"}',tmp_path,_token=lambda *_:'TEST_TOKEN',
        _request=lambda *_:pytest.fail('must not send credential-bearing report'),emit=lambda _:None)
    assert result['reason_code']=='CREDENTIAL_IN_REPORT_REFUSED'


def test_run_retries_observation_failure_without_training_side_effects(fixture,monkeypatch):
    obs,root,home,work,runtime,_=fixture;stop=threading.Event();calls=[]
    def bad(*args,**kwargs):
        calls.append(1);stop.set();raise MemoryError()
    monkeypatch.setattr(obs,'once',bad)
    assert obs.run(root,home,work,stop=stop)==0
    status=json.loads((work/'observer-status.json').read_text())
    assert status['status']=='RETRYING' and status['error_type']=='MemoryError'
    assert calls==[1] and not (runtime/'STOP').exists()


def test_real_detached_start_is_idempotent_and_identity_bound(fixture,monkeypatch):
    obs,root,home,work,runtime,_=fixture
    monkeypatch.setattr(obs,'_command',lambda *a,**k:[sys.executable,'-u','-c','import time;time.sleep(30)'])
    saved=None
    try:
        saved=obs.start(root,home,work)
        assert saved['status']=='STARTING' and os.getsid(saved['pid'])==saved['pid']
        again=obs.start(root,home,work)
        assert again['status']=='ALREADY_RUNNING' and again['pid']==saved['pid']
        assert obs.process_matches(json.loads((work/'observer-process.json').read_text()))
        assert not (runtime/'STOP').exists()
    finally:
        if saved and obs.process_matches(saved):
            import signal
            os.kill(saved['pid'],signal.SIGTERM)
            import time
            until=time.monotonic()+3
            while obs.process_matches(saved) and time.monotonic()<until:time.sleep(.01)


def test_old_monitor_lock_waits_then_can_start_after_release(fixture,monkeypatch):
    obs,root,home,work,runtime,_=fixture;called=[]
    class Stop:
        stopped=False
        def is_set(self):return self.stopped
        def wait(self,seconds):
            assert seconds==5;self.stopped=True
    monkeypatch.setattr(obs,'once',lambda *a,**k:called.append(1))
    with (runtime/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert obs.run(root,home,work,stop=Stop())==0
    assert called==[]
    assert json.loads((work/'observer-status.json').read_text())['status']=='WAITING_LEGACY_MONITOR_EXIT'
    stop=threading.Event()
    def once(*a,**k):called.append(1);stop.set()
    monkeypatch.setattr(obs,'once',once)
    assert obs.run(root,home,work,stop=stop)==0 and called==[1]


def test_retained_result_collection_preserves_summary_protocol_and_skips_same_hash(fixture):
    obs,root,home,work,runtime,_=fixture;results=module('results_observer')
    path=root/'runs/training_campaign_v1/development/abc/receipt.json'
    obs.atomic(path,dict(status='COMPLETE',method='B1',pipeline_seed=20261018,
        participant_equal_ba=.75,publication_wall_seconds=100,chosen_configuration=dict(radius=.1),
        predictions=[1,0],labels=[0,1],token='MUST_NOT_LEAK'))
    obs.atomic(work/'results-observer-request.json',dict(attempt_signal='sig1'))
    calls=[]
    class Client:
        def __init__(self,token,emit,**kwargs):
            assert token=='TEST_TOKEN' and kwargs['branch']=='paper3-results'
        def git_publish(self,files,message):
            calls.append(files)
            assert 'results/paper3/round-01/AUTO_SUMMARY_LATEST.json' in files
            pointer=json.loads(files['results/paper3/round-01/AUTO_SUMMARY_LATEST.json'])
            assert pointer['schema_version']=='paper3-automatic-summary-pointer-v1'
            assert pointer['report_recomputed'] is False
            summaries=[json.loads(v) for k,v in files.items() if k.endswith('/source/runs/training_campaign_v1/development/abc/receipt.json')]
            assert len(summaries)==1
            assert summaries[0]['chosen_configuration']==dict(radius=.1)
            assert summaries[0]['participant_equal_ba']==.75
            assert 'predictions' not in summaries[0] and 'labels' not in summaries[0] and 'token' not in summaries[0]
            return 'c'*40
    before=path.read_bytes()
    status=results.cycle(root,home,work,token_provider=lambda *_:'TEST_TOKEN',client_factory=Client)
    assert status['status']=='SUMMARY_UPLOADED' and path.read_bytes()==before
    assert status['pointer']=='results/paper3/round-01/AUTO_SUMMARY_LATEST.json'
    same=results.cycle(root,home,work,token_provider=lambda *_:pytest.fail('unchanged hash should avoid network'),client_factory=Client)
    assert same['status']=='UNCHANGED_EXISTING_SUMMARY' and len(calls)==1


def test_results_failure_never_changes_scientific_state(fixture):
    obs,root,home,work,runtime,_=fixture;results=module('results_observer')
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    status=results.cycle(root,home,work,collector=lambda *_:(_ for _ in ()).throw(MemoryError()))
    assert status['status']=='RETRYING' and status['error_type']=='MemoryError'
    # Only the independent result lock may have been created in operational metadata.
    after={p:p.read_bytes() for p in root.rglob('*') if p.is_file() and p.name!='auto_summary.lock'}
    assert before==after


def test_results_worker_respects_existing_summary_lock(fixture):
    obs,root,home,work,runtime,_=fixture;results=module('results_observer')
    with (runtime/'auto_summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=results.cycle(root,home,work,collector=lambda *_:pytest.fail('existing summary worker must be preserved'))
    assert status['status']=='EXISTING_RESULTS_WORKER_ACTIVE'


def test_results_dispatcher_is_async_and_only_runs_on_changed_science_signal(fixture):
    obs,root,home,work,runtime,_=fixture;results=module('results_observer')
    now=[1000.];active=[True];calls=[]
    expected=results.command(root,home,work)
    process=lambda pid:dict(pid=333,uid=os.getuid(),argv=expected,start_ticks='10') if pid==333 and active[0] else None
    class Child:
        pid=333
        def poll(self):return None if active[0] else 0
    def popen(argv,**kwargs):
        assert argv==expected and kwargs['start_new_session'] and kwargs['close_fds']
        calls.append(argv);active[0]=True;return Child()
    dispatcher=results.Dispatcher(root,home,work,clock=lambda:now[0],process=process,popen=popen)
    snap=dict(parallel_plan_complete_tasks=245,mainline_complete=False,all_original_complete=False)
    assert dispatcher.tick(snap) and len(calls)==1
    assert not dispatcher.tick(dict(snap,parallel_plan_complete_tasks=246)) and len(calls)==1
    attempt=results.signal_key(snap)
    obs.atomic(work/'results-observer-status.json',dict(status='SUMMARY_UPLOADED',attempt_signal=attempt))
    active[0]=False;now[0]=1601
    assert not dispatcher.tick(snap) and len(calls)==1
    assert dispatcher.tick(dict(snap,parallel_plan_complete_tasks=246)) and len(calls)==2


def test_results_dispatcher_deadline_kills_only_exact_owned_helper(fixture):
    obs,root,home,work,runtime,_=fixture;results=module('results_observer')
    now=[1000.];calls=[];expected=results.command(root,home,work)
    row=dict(pid=333,uid=os.getuid(),argv=expected,start_ticks='10')
    obs.atomic(work/'results-observer-process.json',dict(row,started_unix=0))
    dispatcher=results.Dispatcher(root,home,work,clock=lambda:now[0],process=lambda _:row,
        send=lambda pid,sig:calls.append((pid,sig)),deadline=100)
    assert not dispatcher.tick(dict(parallel_plan_complete_tasks=245))
    import signal
    assert calls==[(333,signal.SIGKILL)]
    # A PID reused by another process has a different start identity and receives nothing.
    calls.clear();dispatcher.process=lambda _:dict(row,start_ticks='11')
    assert not dispatcher.tick(dict(parallel_plan_complete_tasks=245)) and calls==[]


def test_results_signal_does_not_change_for_sampling_heartbeats():
    results=module('results_observer')
    original=dict(parallel_plan_complete_tasks=245,updated_utc='before',mainline_complete=False,
                  extension_arms=dict(Y=dict(status='RUNNING',native_completed_artifact_counts=dict(meta=2))))
    assert results.signal_key(original)==results.signal_key(dict(original,updated_utc='after'))
    changed=dict(original,extension_arms=dict(Y=dict(status='RUNNING',native_completed_artifact_counts=dict(meta=3))))
    assert results.signal_key(original)!=results.signal_key(changed)
    assert results.signal_key(dict(parallel_plan_complete_tasks=None)) is None


def test_native_internal_progress_counts_commits_without_publishing_predictions(tmp_path):
    obs=module('observer');path=tmp_path/'progress.json'
    path.write_text(json.dumps(dict(phase='PREDICTIONS_ONLY',fit_attempts_dispatched=5,
        predictions={'private-person-session-a':dict(path='private prediction.npy')},
        scores={'private-person-session-a':dict(labels=[1,0],ba=.5)},completed={'fit-a':dict(model='private.pt')})))
    selected,meta=obs.progress_values(path)
    assert selected==dict(phase='PREDICTIONS_ONLY',committed_predictions=1,scored_sessions=1,completed_fit_receipts=1)
    assert not meta.get('error')
    assert 'private' not in json.dumps(selected) and 'fit_attempts_dispatched' not in selected


def test_large_internal_progress_uses_strict_stream_counter_projection(tmp_path):
    obs=module('observer');path=tmp_path/'progress.json'
    values={str(i):dict(hidden='x'*131072) for i in range(20)}
    path.write_text(json.dumps(dict(phase='PREDICTIONS_ONLY',predictions=values,scores={'score':dict(hidden='not published')},
                                    completed={},outer_updates_completed=7)))
    assert path.stat().st_size>obs.MAX_JSON
    selected,meta=obs.progress_values(path)
    assert meta['streamed_counter_projection']
    assert selected==dict(phase='PREDICTIONS_ONLY',committed_predictions=20,scored_sessions=1,completed_fit_receipts=0,
                          outer_updates_completed=7)
    assert len(json.dumps(selected))<300


def test_large_truncated_or_duplicate_commits_never_become_completed_counts(tmp_path):
    obs=module('observer');path=tmp_path/'progress.json'
    prefix='{"unused":"'+('x'*(obs.MAX_JSON+1))+'",'
    for ending in ('"predictions":{"a":{},"a":{}}}', '"predictions":{"a":{}'):
        path.write_text(prefix+ending)
        selected,meta=obs.progress_values(path)
        assert selected=={} and meta.get('error')=='ProjectionError'


def test_task_observation_includes_internal_commit_counts(fixture):
    obs,root,home,work,runtime,queue=fixture
    source=root/'runs/training_campaign_v1/progress.json'
    obs.atomic(source,dict(phase='PREDICTIONS_ONLY',predictions={'private-a':{}},scores={}))
    snapshot=obs.collect(root,home,work,runner=queue)
    active=snapshot['mainline']['active_actions'][0]
    assert active['observations'][0]['values']['committed_predictions']==1
    assert active['observations'][0]['values']['scored_sessions']==0
    assert 'private-a' not in json.dumps(snapshot)
