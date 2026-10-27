"""CPU process integration: hand over management while scientific work continues.

No Slurm, GPU, scientific package or network is used. Real /proc identity and
real management-only SIGTERM are exercised with independent deterministic work.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import refresh_hotfix_template as hotfix


MANAGER_SOURCE=r'''import json,os,sys,time
from pathlib import Path
def arg(flag):return sys.argv[sys.argv.index(flag)+1]
work=Path(arg('--work'));work.mkdir(parents=True,exist_ok=True)
record=dict(pid=os.getpid(),launch_token=arg('--launch-token'),
    first_slurm_observation_completed=True,queue_observation_successful=True)
(work/'controller-ready.json').write_text(json.dumps(record))
while True:time.sleep(.02)
'''

SCIENTIFIC_SOURCE=r'''import os,signal,sys,time
from pathlib import Path
folder=Path(sys.argv[1]);count=0
# A management update must not send this independent worker any signal.
def interrupted(signum,frame):
    (folder/'unexpected-signal.json').write_text(str(signum));raise SystemExit(93)
signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
while True:
    count+=1
    stage=folder/'count.tmp';stage.write_text(str(count));os.replace(stage,folder/'count.txt')
    time.sleep(.01)
'''


def wait_until(predicate,timeout=3):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        try:
            value=predicate()
        except (FileNotFoundError,ValueError):value=False
        if value:return value
        time.sleep(.01)
    raise AssertionError('deterministic CPU fixture did not become ready')


def terminate_fixture(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=3)


@pytest.mark.parametrize("first_launch_fails",[False,True])
def test_real_process_management_handover_preserves_continuing_science_and_checkpoint(tmp_path,monkeypatch,first_launch_fails):
    home=tmp_path/'home';root=tmp_path/'project'
    base=home/'paper3-r1-full-repair';old=base/hotfix.OLD_PACKAGE;work=base/'work'
    old.mkdir(parents=True);work.mkdir();root.mkdir()
    old_source=MANAGER_SOURCE.encode();(old/'supervisor.py').write_bytes(old_source)
    replacement=MANAGER_SOURCE.encode();fixed_sha=hashlib.sha256(replacement).hexdigest()
    monkeypatch.setattr(hotfix,'SUPERVISOR_SHA256',fixed_sha)
    # Hash/manifest failure cases have separate real-payload tests. This
    # fixture isolates process ownership, takeover, readiness and no-interrupt.
    monkeypatch.setattr(hotfix,'verified_files',lambda package:{'supervisor.py':replacement})
    science=tmp_path/'independent-science';science.mkdir()
    checkpoint=science/'checkpoint.bin';checkpoint.write_bytes(b'committed-checkpoint-v1\x00preserve')
    done=science/'receipt.json';done.write_text('{"status":"COMPLETE","scientific_identity":"same"}\n')
    expected_checkpoint=checkpoint.read_bytes();expected_receipt=done.read_bytes()
    old_manifest=b'original immutable package fixture\n';(old/'MANIFEST.json').write_bytes(old_manifest)
    original_states=root/'runs/training_campaign_parallel_v1';original_states.mkdir(parents=True)
    schedule=original_states/'scheduler-state.json'
    schedule_bytes=json.dumps({'slots':{str(i):{'attempts':[{'job_id':job,'slurm_state':'RUNNING'}]}
        for i,job in enumerate(['772906','772904','772905'])}},sort_keys=True).encode()
    schedule.write_bytes(schedule_bytes)
    science_script=science/'worker_fixture.py';science_script.write_text(SCIENTIFIC_SOURCE)
    worker=manager=None;new_pid=None;sent=[]
    try:
        worker=subprocess.Popen([sys.executable,'-I','-S','-B',str(science_script),str(science)],
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        manager_command=[sys.executable,'-I','-S','-B','-u',str(old/'supervisor.py'),'manage',
            '--root',str(root),'--home',str(home),'--package',str(old),'--work',str(work),
            '--submit','--launch-token','original-management-nonce','--python',sys.executable]
        manager=subprocess.Popen(manager_command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,start_new_session=True)
        wait_until(lambda:(work/'controller-ready.json').exists())
        initial=wait_until(lambda:int((science/'count.txt').read_text())>=2)
        old_identity=wait_until(lambda:hotfix.identity(manager.pid))
        record=dict(pid=manager.pid,root=str(root),launch_token='original-management-nonce',
            start_ticks=old_identity['start_ticks'])
        (work/'controller-process.json').write_text(json.dumps(record))
        before=int((science/'count.txt').read_text())
        def only_management_signal(pid,signum):
            sent.append((pid,signum))
            assert pid==manager.pid and pid!=worker.pid
            assert signum==signal.SIGTERM
            os.kill(pid,signum)
        if first_launch_fails:
            def unavailable(*a,**kw):raise OSError('deterministic process resource failure')
            with pytest.raises(OSError,match='deterministic'):
                hotfix.apply(home,root,activate=True,send=only_management_signal,popen=unavailable)
            # An installation retry must recover the recorded management stop
            # while the already-running scientific work continues normally.
            assert worker.poll() is None
            assert checkpoint.read_bytes()==expected_checkpoint
            assert int((science/'count.txt').read_text())>=before
        value=hotfix.apply(home,root,activate=True,send=only_management_signal)
        new_pid=value['manager_pid']
        assert value['status']=='R1_REFRESH_HOTFIX_CONTROLLER_READY'
        assert new_pid not in (manager.pid,worker.pid)
        assert sent==[(manager.pid,signal.SIGTERM)]
        assert wait_until(lambda:int((science/'count.txt').read_text())>before)
        assert worker.poll() is None
        assert not (science/'unexpected-signal.json').exists()
        assert checkpoint.read_bytes()==expected_checkpoint
        assert done.read_bytes()==expected_receipt
        assert schedule.read_bytes()==schedule_bytes
        assert (old/'supervisor.py').read_bytes()==old_source
        assert (old/'MANIFEST.json').read_bytes()==old_manifest
        assert value['scientific_children_signalled']==0
        assert value['allocations_cancelled']==0
        again=hotfix.apply(home,root,activate=True,send=lambda *a:pytest.fail('repeat application sent a signal'))
        assert again['status']=='R1_REFRESH_HOTFIX_ALREADY_RUNNING'
        assert again['manager_pid']==new_pid and again['controller_ready'] is True
        assert worker.poll() is None and int((science/'count.txt').read_text())>before
    finally:
        # Teardown signals only the local CPU fixtures after all invariants.
        if new_pid and hotfix.identity(new_pid) is not None:
            try:os.kill(new_pid,signal.SIGTERM)
            except ProcessLookupError:pass
        terminate_fixture(manager);terminate_fixture(worker)
