"""Driver tests use real detached stdlib processes and generated adapters only."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

HERE=Path(__file__).resolve().parent


def module():
    spec=importlib.util.spec_from_file_location('_result_export_driver_test',HERE/'delivery.py')
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def wait_for(predicate,seconds=5):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        value=predicate()
        if value:return value
        time.sleep(.02)
    raise AssertionError('Expected process state was not observed')


ADAPTER='''
from pathlib import Path
import json,os,sys,time
from delivery import atomic,read
def remember(args):
 assert 'torch' not in sys.modules and 'numpy' not in sys.modules
 path=Path(args.work)/(args.role+'-calls.json')
 rows=read(path,[]) or []
 rows.append(dict(pid=os.getpid(),role=args.role))
 atomic(path,rows)
 return len(rows)
def paper_once(args):
 remember(args)
 secret=Path(args.work)/'raise-sensitive-error'
 if secret.exists(): raise RuntimeError(secret.read_text())
 while not (Path(args.work)/('release-'+args.role)).exists() and not (Path(args.work)/'STOP').exists():time.sleep(.01)
 return dict(status='AVAILABLE_PAPER_UPLOADED',paper_ready=True,complete=True)
def full_once(args):
 n=remember(args)
 assert all((read(Path(args.work)/role/'status.json',{}) or {}).get('priority_ready') for role in ('r1-paper','r11-paper'))
 return dict(status='COMPLETE' if n==2 else 'UPLOADING',complete=n==2,group=n)
def legacy_uploader_candidates(args,role):
 assert role=='full'
 return []
'''


@pytest.fixture
def cloned(tmp_path):
    bundle=tmp_path/'bundle';bundle.mkdir()
    source=(HERE/'delivery.py').read_text()
    # Exercise unchanged control logic without waiting the production polling periods.
    for seconds in (5,10,20,30):source=source.replace('time.sleep('+str(seconds)+')','time.sleep(.02)')
    (bundle/'delivery.py').write_text(source)
    for name in ('r1_export.py','r11_export.py'):(bundle/name).write_text(ADAPTER)
    (bundle/'torch.py').write_text('raise AssertionError("No Torch imports allowed")\n')
    root=tmp_path/'project';root.mkdir();work=tmp_path/'work';work.mkdir()
    protected=[]
    for relative in ('runs/training_campaign_parallel_v1/STOP','runs/r11_campaign/runtime/STOP'):
        p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'explicit-scientific-stop\n');protected.append(p)
    args=['--root',str(root),'--work',str(work),'--plan',str(root/'runs/r11_campaign/plan.json')]
    return bundle,root,work,args,protected


def run(cloned,*extra):
    bundle,root,work,args,_=cloned
    return subprocess.run([sys.executable,'-I','-S','-B','-u',str(bundle/'delivery.py'),*extra,*args],
        capture_output=True,text=True,timeout=12)


def final_json(result):
    return json.loads(result.stdout.splitlines()[-1])


def test_real_isolated_check_is_readonly_and_does_not_import_adapters_or_torch(cloned):
    _,root,work,args,protected=cloned
    before={str(p):p.read_bytes() for p in protected}
    result=run(cloned,'--check')
    assert result.returncode==0,result.stderr
    row=final_json(result)
    assert row['event']=='PAPER3_R1_R11_EXPORT_STATUS'
    assert all(value['status']=={} for value in row['roles'].values())
    assert not list(work.glob('*-calls.json'))
    assert {str(p):p.read_bytes() for p in protected}==before


def test_four_detached_roles_survive_launcher_reuse_and_gate_full_until_both_paper_ready(cloned):
    driver=module();_,root,work,args,protected=cloned
    original={str(p):p.read_bytes() for p in protected}
    pids=[]
    try:
        result=run(cloned,'--start');assert result.returncode==0,result.stdout+result.stderr
        first=final_json(result)
        assert first['terminal_can_close'] is True and first['scientific_stops_retained'] is True
        assert first['slurm_jobs_submitted']==first['slurm_jobs_cancelled']==0
        pids=[row['pid'] for row in first['roles']]
        assert len(set(pids))==4
        for pid in pids:
            assert driver.identity(pid) is not None
            assert os.getsid(pid)==pid
        wait_for(lambda:all((work/(role+'-calls.json')).exists() for role in ('r1-paper','r11-paper')))
        second=run(cloned,'--start');assert second.returncode==0
        rows=final_json(second)['roles']
        assert all(row['status']=='EXISTING_BACKGROUND_WORKER_REUSED' for row in rows)
        assert [row['process']['pid'] for row in rows]==pids
        assert not (work/'r1-full-calls.json').exists() and not (work/'r11-full-calls.json').exists()
        (work/'release-r1-paper').touch()
        wait_for(lambda:(driver.read(work/'r1-paper/status.json',{}) or {}).get('priority_ready'))
        time.sleep(.05)
        assert not (work/'r1-full-calls.json').exists() and not (work/'r11-full-calls.json').exists()
        (work/'release-r11-paper').touch()
        wait_for(lambda:all((driver.read(work/role/'status.json',{}) or {}).get('complete') for role in driver.ROLES))
        for role in ('r1-full','r11-full'):
            calls=driver.read(work/(role+'-calls.json'))
            assert len(calls)==2 and len({r['pid'] for r in calls})==2
        assert {str(p):p.read_bytes() for p in protected}==original
        assert not (work/'STOP').exists()
    finally:
        (work/'STOP').touch()
        # Stub children observe only their own upload STOP and exit; no signals
        # are sent to any scientific process or process group in this cleanup.
        until=time.monotonic()+5
        while any(driver.identity(pid) for pid in pids) and time.monotonic()<until:time.sleep(.02)


@pytest.mark.parametrize('role,module_name,method',[('r1-paper','r1_export','paper_once'),
    ('r11-paper','r11_export','paper_once'),('r1-full','r1_export','full_once'),('r11-full','r11_export','full_once')])
def test_exact_round_and_channel_adapter_routing(tmp_path,monkeypatch,role,module_name,method):
    driver=module();calls=[]
    fake=SimpleNamespace(paper_once=lambda args:calls.append('paper_once') or dict(status='OK',paper_ready=True),
        full_once=lambda args:calls.append('full_once') or dict(status='OK',complete=True))
    def importer(name):assert name==module_name;return fake
    monkeypatch.setattr(driver.importlib,'import_module',importer)
    args=SimpleNamespace(role=role,work=str(tmp_path))
    result=driver.once(args)
    assert calls==[method] and result['role']==role
    assert result['slurm_jobs_submitted']==result['slurm_jobs_cancelled']==0
    if method=='paper_once':assert result['priority_ready'] is True


@pytest.mark.parametrize('change',('pid','uid','argv','start_ticks'))
def test_reused_or_foreign_uploader_identity_is_never_signalled(change):
    driver=module();row=dict(pid=303,uid=os.getuid(),argv=['python','upload','--channel','full'],start_ticks='100',role='full',kind='uploader')
    current={k:row[k] for k in ('pid','uid','argv','start_ticks')}
    current[change]=['other'] if change=='argv' else 'different'
    sent=[]
    result=driver.stop_uploader(row,process=lambda _:current,send=lambda *v:sent.append(v))
    assert result['signals']==[] and sent==[]


@pytest.mark.parametrize('change',({'role':'paper'},{'kind':'worker'},{'uid':-1}))
def test_non_full_cpu_uploader_role_is_refused(change):
    driver=module();row=dict(pid=303,uid=os.getuid(),argv=['python'],start_ticks='100',role='full',kind='uploader');row.update(change)
    sent=[]
    with pytest.raises(ValueError,match='EXACT_FULL_UPLOADER_ROLE_REQUIRED'):
        driver.stop_uploader(row,send=lambda *values:sent.append(values))
    assert not sent


def test_real_cpu_uploader_handoff_leaves_same_group_scientific_counter_running(tmp_path):
    driver=module();counter=tmp_path/'counter'
    upload=subprocess.Popen([sys.executable,'-u','-c','import time;print("ready",flush=True);time.sleep(30)'],stdout=subprocess.PIPE,text=True)
    science=subprocess.Popen([sys.executable,'-u','-c','import sys,time;from pathlib import Path;p=Path(sys.argv[1]);n=0\nwhile True:\n n+=1;p.write_text(str(n));time.sleep(.01)',str(counter)])
    try:
        assert upload.stdout.readline().strip()=='ready'
        wait_for(counter.exists)
        assert os.getpgid(upload.pid)==os.getpgid(science.pid)
        row=dict(driver.identity(upload.pid),role='full',kind='uploader')
        result=driver.stop_uploader(row)
        upload.wait(timeout=3)
        assert result['signals']==['SIGTERM'] and result['scientific_processes_signalled']==0
        assert science.poll() is None
        first=counter.read_text();wait_for(lambda:counter.read_text()!=first)
    finally:
        for child in (upload,science):
            if child.poll() is None:child.terminate()
            child.wait(timeout=3)


def test_bounded_corrupt_metadata_is_not_silently_reset(tmp_path):
    driver=module();p=tmp_path/'state.json';p.write_text('{"incomplete":')
    with pytest.raises(ValueError):driver.read(p,{})
    p.write_bytes(b'x'*1025)
    with pytest.raises(ValueError,match='EXPORT_METADATA_TOO_LARGE'):driver.read(p,{},limit=1024)


def test_sensitive_error_body_is_not_logged_even_when_uppercase(cloned):
    _,root,work,args,_=cloned
    secret='GITHUB_SECRET_TOKEN_123'
    (work/'raise-sensitive-error').write_text(secret)
    result=run(cloned,'--once','--role','r1-paper')
    assert result.returncode==1
    assert secret not in result.stdout+result.stderr
    assert secret not in (work/'r1-paper/status.json').read_text()
    assert final_json(result)['error_type']=='RuntimeError'
