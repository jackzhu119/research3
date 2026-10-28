import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))


def load():
    spec=importlib.util.spec_from_file_location('_single_launch_test',HERE/'launch.py')
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


@pytest.fixture
def layout(tmp_path):
    module=load();home=tmp_path/'home';root=home/'projects/paper3';work=home/'single/work'
    native=home/'native';native.mkdir(parents=True)
    runtime=root/module.OPERATIONAL;runtime.mkdir(parents=True)
    old=home/'paper3-r1-full-repair/2bfd664251ee5d1bcaf5';old.mkdir(parents=True)
    for name in ('worker.py','kernel_bridge.py','action_startup.py','START.py'):(native/name).write_text('# retained\n')
    (old/'parallel_runtime.py').write_text('# existing metadata adapter\n')
    (old/'verified_context_cache.py').write_text('# existing action cache\n')
    argv=[sys.executable,'-I','-S','-B','-u',str(native/'worker.py'),'--root',str(root),
          '--runtime-dir',str(runtime),'--slot','{slot}','--package',str(native),
          '--payload-manifest',str(home/'manifest.json'),'--direct-package',str(home/'direct')]
    module.atomic(runtime/'launch.json',dict(root=str(root),python=sys.executable,package=str(native),worker_command=argv))
    return module,home,root,work,native,runtime,old


def test_context_preserves_native_arguments_and_uses_isolated_metadata_parent(layout):
    launch,home,root,work,native,runtime,old=layout
    config=launch.context(root,home,work)
    oldcmd=json.loads((runtime/'launch.json').read_text())['worker_command']
    cmd=config['worker_command'];at=oldcmd.index('--root')
    assert cmd[:5]==[sys.executable,'-I','-S','-B','-u']
    assert cmd[5]==str(HERE/'worker_runner.py') and cmd[6:]==oldcmd[at:]
    assert config['native_package']==str(native) and config['legacy_dir']==str(old)
    assert config['resources']['gpus_total']==1 and config['resources']['cpus_total']==4


def test_context_refuses_foreign_root_or_worker_script(layout):
    launch,home,root,work,native,runtime,old=layout
    saved=json.loads((runtime/'launch.json').read_text());saved['root']='/other'
    launch.atomic(runtime/'launch.json',saved)
    with pytest.raises(ValueError,match='ORIGINAL_R1_LAUNCH_REQUIRED'):launch.context(root,home,work)
    saved['root']=str(root);saved['worker_command'][5]='/foreign/worker.py';launch.atomic(runtime/'launch.json',saved)
    with pytest.raises(ValueError,match='COMMAND_LAYOUT_DIFFERS'):launch.context(root,home,work)


def test_legacy_identity_reuse_and_wrong_roles_are_never_signalled(layout):
    launch,home,root,work,*_=layout;oldwork=home/'paper3-r1-full-repair/work';oldwork.mkdir(parents=True)
    actual=dict(pid=999,start_ticks='new',uid=os.getuid(),argv=[sys.executable,str(home/'paper3-r1-controller-recovery/v/recovery.py'),
        '--guard','--root',str(root),'--home',str(home),'--token','NEW'])
    launch.atomic(oldwork/'memory-recovery-process.json',dict(actual,start_ticks='old'))
    stopped=[]
    config=dict(root=str(root),home=str(home),work=str(work))
    assert launch.stop_prior_managers(config,process=lambda _:actual,stop=lambda r:stopped.append(r))==[]
    assert not stopped
    launch.atomic(oldwork/'memory-recovery-process.json',actual)
    foreign=dict(actual,argv=[sys.executable,'/other/worker.py','--root',str(root),'--child'])
    assert launch.stop_prior_managers(config,process=lambda _:foreign,stop=lambda r:stopped.append(r))==[]
    assert not stopped


def test_exact_recorded_cpu_manager_is_eligible_for_handoff(layout):
    launch,home,root,work,*_=layout;oldwork=home/'paper3-r1-full-repair/work';oldwork.mkdir(parents=True)
    actual=dict(pid=999,start_ticks='same',uid=os.getuid(),argv=[sys.executable,str(home/'paper3-r1-controller-recovery/v/recovery.py'),
        '--guard','--root',str(root),'--home',str(home),'--token','same-nonce'])
    launch.atomic(oldwork/'memory-recovery-process.json',actual);seen=[]
    result=launch.stop_prior_managers(dict(root=str(root),home=str(home),work=str(work)),process=lambda _:actual,
        stop=lambda record:seen.append(record) or dict(status='STOPPED'))
    assert seen==[actual] and result==[dict(status='STOPPED')]


def test_stop_exact_only_signals_same_identity_and_not_process_group():
    launch=load();manager=subprocess.Popen([sys.executable,'-u','-c','import time;print("ready",flush=True);time.sleep(30)'],stdout=subprocess.PIPE,text=True)
    scientific=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    try:
        assert manager.stdout.readline().strip()=='ready'
        current=launch.identity(manager.pid);assert current
        assert os.getpgid(manager.pid)==os.getpgid(scientific.pid)
        result=launch.stop_exact(current)
        manager.wait(timeout=3)
        assert result['status']=='MANAGEMENT_STOPPED' and result['signals']==['SIGTERM']
        assert scientific.poll() is None
    finally:
        for child in (manager,scientific):
            if child.poll() is None:child.terminate()
            child.wait(timeout=3)


def test_stop_exact_refuses_reused_pid_before_signal():
    launch=load();row=dict(pid=333,argv=['correct'],start_ticks='1');sent=[]
    result=launch.stop_exact(row,process=lambda _:dict(row,start_ticks='2'),send=lambda *a:sent.append(a))
    assert result['signals']==[] and not sent


def cloned_guard_bundle(tmp_path):
    bundle=tmp_path/'bundle';bundle.mkdir()
    (bundle/'launch.py').write_text((HERE/'launch.py').read_text())
    (bundle/'single_gpu.py').write_text('''
import argparse,json,os,time
from pathlib import Path
OPERATIONAL='runs/training_campaign_parallel_v1'
def read(path,default=None):
 try:return json.loads(Path(path).read_text())
 except FileNotFoundError:return default
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+'.tmp-'+str(os.getpid()));temp.write_text(json.dumps(value));os.replace(temp,path)
def emit(event,**fields):print(json.dumps(dict(event=event,**fields)),flush=True)
def resources():return dict(jobs=1,gpus_total=1,cpus_total=4,memory_total='64G')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--result',required=True);a,_=p.parse_known_args()
 atomic(a.result,dict(status='RUNNING',phase='FAKE_METADATA_ONLY',scheduler=dict(jobs=[],inflight_jobs=0),all_original_complete=False))
''')
    (bundle/'observer.py').write_text('''
def legacy_monitor_candidates(*args):return []
def start(*args):return dict(status='STARTING',test_only=True)
''')
    return bundle


def wait_file(path,timeout=5):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        try:return json.loads(path.read_text())
        except (FileNotFoundError,ValueError):time.sleep(.02)
    raise AssertionError('missing '+str(path))


def test_actual_detached_guardian_duplicate_apply_and_science_counter_continues(layout,tmp_path):
    launch,home,root,work,*_=layout;bundle=cloned_guard_bundle(tmp_path)
    counter=tmp_path/'scientific-counter'
    code='''import pathlib,sys,time
p=pathlib.Path(sys.argv[1]);n=0
while True:
 n+=1;p.write_text(str(n));time.sleep(.01)
'''
    science=subprocess.Popen([sys.executable,'-u','-c',code,str(counter)])
    args=[sys.executable,'-I','-S','-B','-u',str(bundle/'launch.py'),'--apply','--root',str(root),'--home',str(home),'--work',str(work)]
    record=None
    try:
        a=subprocess.run(args,capture_output=True,text=True,timeout=8)
        assert a.returncode==0,a.stderr
        record=wait_file(work/'controller-process.json');pid=record['pid']
        assert os.getsid(pid)==pid
        status=wait_file(work/'controller-status.json')
        assert status['phase']=='FAKE_METADATA_ONLY'
        b=subprocess.run(args,capture_output=True,text=True,timeout=8)
        assert b.returncode==0,b.stderr
        assert 'ALREADY_RUNNING' in b.stdout
        assert wait_file(work/'controller-process.json')['pid']==pid
        while not counter.exists():time.sleep(.01)
        before=int(counter.read_text())
        os.kill(pid,signal.SIGTERM)
        until=time.monotonic()+4
        while launch.identity(pid) and time.monotonic()<until:time.sleep(.02)
        assert launch.identity(pid) is None
        time.sleep(.05)
        assert science.poll() is None and int(counter.read_text())>before
        assert not (root/launch.OPERATIONAL/'STOP').exists()
    finally:
        if record and launch.identity(record['pid']):os.kill(record['pid'],signal.SIGKILL)
        science.terminate();science.wait(timeout=3)


def test_actual_guard_honors_durable_stop_without_cycle_or_science_signal(layout,tmp_path):
    launch,home,root,work,native,runtime,old=layout;bundle=cloned_guard_bundle(tmp_path)
    config=launch.context(root,home,work);launch.atomic(work/'config.json',config)
    (runtime/'STOP').touch()
    args=[sys.executable,'-I','-S','-B','-u',str(bundle/'launch.py'),'--guard','--root',str(root),
        '--home',str(home),'--work',str(work),'--token','test-stop']
    result=subprocess.run(args,capture_output=True,text=True,timeout=8)
    assert result.returncode==0,result.stderr
    status=wait_file(work/'controller-status.json')
    assert status['status']=='STOPPED_BY_USER' and (runtime/'STOP').exists()
    assert not list(work.glob('cycle-*.json'))
