from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import supervisor as s
import processes as p
from atomic_io import atomic_bytes


HELPER = '''import fcntl,json,os,signal,sys,time
from pathlib import Path
args=sys.argv[1:];mode=args[0]
def arg(name):return args[args.index(name)+1] if name in args else None
plan=json.loads(Path(arg('--plan')).read_text());runtime=Path(plan['runtime'])
role=('report' if mode=='report-loop' else arg('--channel') if mode=='upload' else mode)
lock=runtime/('controller.lock' if role=='controller' else role+'-upload.lock')
handles=[];scope='original_upload';observer=False
handle=lock.open('a');handles.append(handle)
try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
 if role not in ('paper','full'):
  while True:time.sleep(.05)
 observer=True;scope='repair_observer'
 handle=(Path(arg('--work'))/(role+'-observer.lock')).open('a');handles.append(handle);fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
if arg('--work'):
 work=Path(arg('--work'));work.mkdir(parents=True,exist_ok=True)
 ready=dict(status='READY',role=role,pid=os.getpid(),launch_token=arg('--launch-token'),recipe_sha256=plan['recipe_sha256'],lock_acquired=True,
   first_slurm_observation_completed=True,queue_observation_successful=True,lock_scope=scope,
   original_upload_lock_owned=not observer,existing_uploader_preserved=observer)
 (work/(role+'-ready.json')).write_text(json.dumps(ready))
signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))
while True:time.sleep(.05)
'''


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def study(tmp_path,monkeypatch):
    package=tmp_path/'native';native=package/'paper3r11/platform/entry.py';native.parent.mkdir(parents=True);native.write_text(HELPER)
    new=tmp_path/'new';new.mkdir();(new/'entry.py').write_text(HELPER)
    monkeypatch.setattr(s,'HERE',new);monkeypatch.setattr(p,'HERE',new)
    runtime=tmp_path/'runtime';runtime.mkdir();work=tmp_path/'repair-work'
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(dict(runtime=str(runtime),recipe_sha256='f'*64)))
    scientific=tmp_path/'actual-scientific-result.npz';scientific.write_bytes(b'original sealed science bytes')
    def atomic(path,value):atomic_bytes(path,(json.dumps(value,sort_keys=True)+'\n').encode())
    env=dict(path=plan,package=package,runtime=runtime,
             common=SimpleNamespace(sha=sha,atomic=atomic),
             header=dict(plan_sha256=sha(plan),recipe_sha256='f'*64,python=sys.executable,total_tasks=10270))
    args=SimpleNamespace(submit=True,plan=plan,package=package,work=work,python=sys.executable,credentials=None)
    monkeypatch.setattr(s,'load_env',lambda args:env)
    monkeypatch.setattr(s,'validate_operational_history',lambda env:dict(known_job_ids=['776301','776302','775288','775289']))
    children=[]
    def popen(command,**kwargs):
        child=subprocess.Popen(command,**kwargs);children.append(child);return child
    yield env,args,popen,children,scientific
    for child in children:
        if child.poll() is None:child.terminate()
    for child in children:
        try:child.wait(timeout=4)
        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=2)  # fixture processes only


def native(study,role):
    env,args,popen,children,_=study
    tail=(['report-loop'] if role=='report' else ['upload','--channel',role] if role!='controller' else ['controller'])
    child=popen([sys.executable,'-I','-S','-B','-u',str(env['package']/'paper3r11/platform/entry.py'),*tail,
                 '--plan',str(env['path'])],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    until=time.monotonic()+4
    while time.monotonic()<until and not p.lock_live(s._role_lock(env,role)):time.sleep(.01)
    assert p.lock_live(s._role_lock(env,role))
    return child


def test_real_detached_five_roles_start_idempotently_without_science_changes(study):
    env,args,popen,children,scientific=study
    before={str(path):path.read_bytes() for path in [scientific,env['path'],env['package']/'paper3r11/platform/entry.py']}
    first=s.start(args,popen=popen,ready_timeout=3)
    assert first['status']=='MANAGEMENT_REPAIR_READY' and first['ready_helpers']==5
    pids=[row['pid'] for row in first['processes']]
    second=s.start(args,popen=popen,ready_timeout=3)
    assert second['status']=='MANAGEMENT_REPAIR_READY'
    assert [row['pid'] for row in second['processes']]==pids and len(children)==5
    assert first['scientific_workers_signalled']==first['slurm_jobs_cancelled']==0
    assert before=={str(path):path.read_bytes() for path in [scientific,env['path'],env['package']/'paper3r11/platform/entry.py']}


def test_healthy_full_and_paper_writers_preserved_with_observers(study):
    env,args,popen,children,_=study
    full=native(study,'full');paper=native(study,'paper')
    legacy=env['runtime']/'full-upload-process.json';legacy.write_text(json.dumps(dict(pid=full.pid,legacy=True)))
    before=legacy.read_bytes()
    result=s.start(args,popen=popen,ready_timeout=3)
    assert result['status']=='MANAGEMENT_REPAIR_READY'
    assert full.poll() is None and paper.poll() is None and legacy.read_bytes()==before
    for row in result['processes']:
        if row['role'] in ('full','paper'):
            assert row['status'].endswith('OBSERVER_READY_LEGACY_UPLOADER_PRESERVED')
            assert row['old_helpers_signalled']==0 and row['original_upload_lock_owned'] is False


def test_only_owned_old_management_stops_and_lock_releases_before_ready(study):
    env,args,popen,children,_=study
    old=native(study,'controller');old_monitor=native(study,'monitor')
    result=s.start(args,popen=popen,ready_timeout=3)
    assert result['status']=='MANAGEMENT_REPAIR_READY'
    old.wait(timeout=3);old_monitor.wait(timeout=3)
    rows={row['role']:row for row in result['processes']}
    assert rows['controller']['handover'][0]['signals']==['SIGTERM']
    assert rows['monitor']['handover'][0]['signals']==['SIGTERM']
    assert p.lock_live(env['runtime']/'controller.lock')


def test_lost_process_receipts_discovered_without_duplicate_start(study):
    env,args,popen,children,_=study
    first=s.start(args,popen=popen,ready_timeout=3);assert first['status']=='MANAGEMENT_REPAIR_READY'
    for path in args.work.glob('*-process.json'):path.unlink()
    for path in env['runtime'].glob('*-process.json'):path.unlink()
    second=s.start(args,popen=popen,ready_timeout=3)
    assert [row['pid'] for row in second['processes']]==[row['pid'] for row in first['processes']]
    assert len(children)==5


def test_one_helper_launch_failure_does_not_block_other_roles(study):
    env,args,popen,children,_=study
    def fail_controller(command,**kwargs):
        if command[6]=='controller':raise OSError('injected helper launch failure')
        return popen(command,**kwargs)
    result=s.start(args,popen=fail_controller,ready_timeout=3)
    assert result['status']=='MANAGEMENT_REPAIR_PARTIAL'
    assert result['processes'][0]['status']=='HELPER_START_RETRY_REQUIRED'
    assert result['ready_helpers']==4 and len(children)==4


def test_startup_pending_nonce_adopted_without_duplicate_spawn(study,monkeypatch):
    env,args,popen,children,_=study
    original=s.ready_for
    monkeypatch.setattr(s,'ready_for',lambda *args:None)
    first=s.start(args,popen=popen,ready_timeout=.2)
    assert first['status']=='MANAGEMENT_REPAIR_STARTUP_PENDING'
    time.sleep(.15)
    second=s.start(args,popen=popen,ready_timeout=.2)
    assert second['status']=='MANAGEMENT_REPAIR_STARTUP_PENDING' and len(children)==5
    monkeypatch.setattr(s,'ready_for',original)
    assert s.start(args,popen=popen,ready_timeout=3)['status']=='MANAGEMENT_REPAIR_READY'


def test_start_requires_explicit_submit_and_check_does_not_write(study):
    env,args,popen,children,_=study
    args.submit=False
    with pytest.raises(ValueError,match='explicit --submit'):s.start(args,popen=popen)
    assert not args.work.exists() and children==[]
    before={str(path):path.read_bytes() for path in env['runtime'].rglob('*') if path.is_file()}
    checked=s.check(args)
    assert checked['status']=='READ_ONLY_MANAGEMENT_CHECK' and checked['slurm_calls']==0
    assert not args.work.exists()
    assert before=={str(path):path.read_bytes() for path in env['runtime'].rglob('*') if path.is_file()}


def test_prior_error_bytes_preserved_before_controller_replacement(study):
    env,args,popen,children,_=study
    previous=b'{"error_type":"ValueError","reason":"operational state unreadable"}\n'
    (env['runtime']/'controller-error.json').write_bytes(previous)
    assert s.start(args,popen=popen,ready_timeout=3)['status']=='MANAGEMENT_REPAIR_READY'
    saved=list((args.work/'history/controller-error').glob('*.json'))
    assert len(saved)==1 and saved[0].read_bytes()==previous


def test_unknown_controller_lock_owner_never_signalled_and_other_roles_start(study):
    env,args,popen,children,_=study
    lock=env['runtime']/'controller.lock'
    code="import fcntl,time,sys;h=open(sys.argv[1],'a');fcntl.flock(h,fcntl.LOCK_EX);time.sleep(30)"
    unrelated=popen([sys.executable,'-c',code,str(lock)],stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    until=time.monotonic()+3
    while time.monotonic()<until and not p.lock_live(lock):time.sleep(.01)
    assert p.lock_live(lock)
    result=s.start(args,popen=popen,ready_timeout=3)
    assert result['status']=='MANAGEMENT_REPAIR_PARTIAL'
    assert result['processes'][0]['status']=='LOCK_OWNER_UNCONFIRMED_LEFT_UNTOUCHED'
    assert result['ready_helpers']==4 and unrelated.poll() is None


def test_delayed_new_process_identity_is_pending_then_adopted(study):
    env,args,popen,children,_=study
    first=s.start(args,popen=popen,ready_timeout=0)
    assert first['status'] in ('MANAGEMENT_REPAIR_STARTUP_PENDING','MANAGEMENT_REPAIR_READY')
    time.sleep(.2)
    second=s.start(args,popen=popen,ready_timeout=3)
    assert second['status']=='MANAGEMENT_REPAIR_READY' and len(children)==5


def test_bad_history_does_not_prevent_unrelated_observers(study,monkeypatch):
    env,args,popen,children,_=study
    def broken(env):raise OSError('transient history read')
    monkeypatch.setattr(s,'validate_operational_history',broken)
    result=s.start(args,popen=popen,ready_timeout=3)
    assert result['status']=='MANAGEMENT_REPAIR_READY'
    observed=json.loads((args.work/'INPUT.json').read_text())['history']
    assert observed['prior_authorization_verified'] is False and observed['controller_must_revalidate'] is True


def test_real_scientific_worker_survives_management_handover(study):
    env,args,popen,children,scientific=study
    worker=popen([sys.executable,'-I','-S','-B','-u',str(env['package']/'paper3r11/platform/entry.py'),
                  'worker','--plan',str(env['path'])],stdin=subprocess.DEVNULL,
                 stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    time.sleep(.03)
    old=native(study,'controller')
    before=scientific.read_bytes()
    result=s.start(args,popen=popen,ready_timeout=3)
    assert result['status']=='MANAGEMENT_REPAIR_READY'
    old.wait(timeout=3)
    assert worker.poll() is None and scientific.read_bytes()==before
    assert result['scientific_workers_signalled']==0


def test_scientific_cli_arguments_forwarded_without_management_reparse(monkeypatch):
    seen=[]
    fake=SimpleNamespace(science_main=lambda argv:(seen.append(argv) or 0))
    monkeypatch.setitem(sys.modules,'controller',fake)
    original=['science-daemon','--plan','/exact/plan.json','--package','/exact/package',
              '--device','cuda:0','--response-dir','/responses','--future-native-option','literal value']
    assert s.main(original)==0
    assert seen==[original]


def test_worker_once_and_device_are_preserved(monkeypatch):
    seen=[]
    def worker(path,slot,**kwargs):seen.append((path,slot,kwargs));return 75
    monkeypatch.setitem(sys.modules,'controller',SimpleNamespace(worker_loop=worker))
    assert s.main(['worker','--plan','/p','--package','/pkg','--slot','3','--device','cuda:0','--once'])==75
    assert seen==[(Path('/p'),3,dict(package=Path('/pkg'),device='cuda:0',once=True))]
