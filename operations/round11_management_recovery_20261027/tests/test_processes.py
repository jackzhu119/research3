from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import processes as p


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def env(tmp_path):
    package = tmp_path/'original'; entry = package/'paper3r11/platform/entry.py'
    entry.parent.mkdir(parents=True)
    entry.write_text('import time\ntime.sleep(30)\n')
    plan = tmp_path/'plan.json'; plan.write_text('{"recipe_sha256":"fixed"}')
    runtime = tmp_path/'runtime';runtime.mkdir()
    return dict(path=plan,package=package,runtime=runtime,
                common=SimpleNamespace(sha=checksum),
                header=dict(plan_sha256=checksum(plan),python=sys.executable,recipe_sha256='fixed'))


def spawn(env, mode, tail=()):
    return subprocess.Popen([sys.executable, '-I','-S','-B','-u',
        str(env['package']/'paper3r11/platform/entry.py'),mode,'--plan',str(env['path']),*tail],
        stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


def await_identity(child, env, role):
    until=time.monotonic()+3
    while time.monotonic()<until:
        value=p.identify(child.pid,env,role)
        if value:return value
        if child.poll() is not None:break
        time.sleep(.01)
    return None


def cleanup(child):
    if child.poll() is None:child.terminate()
    child.wait(timeout=4)


@pytest.mark.parametrize('role,mode,tail',[
    ('controller','controller',()),('report','report-loop',()),
    ('paper','upload',('--channel','paper')),('full','upload',('--channel','full')),
    ('monitor','upload',('--channel','monitor'))])
def test_exact_original_management_roles_match_real_process(env, role, mode, tail):
    child=spawn(env,mode,tail)
    try:
        row=await_identity(child,env,role)
        assert row and row['pid']==child.pid and row['role']==role
        assert row['current_patch'] is False
        assert p.identify(child.pid,env,'worker') is None
    finally:cleanup(child)


def test_worker_science_and_report_child_never_match_management(env):
    for mode in ('worker','science','science-daemon','report'):
        child=spawn(env,mode)
        try:
            time.sleep(.03)
            assert all(p.identify(child.pid,env,role) is None for role in p.ROLES)
            assert child.poll() is None
        finally:cleanup(child)


def test_plan_and_duplicate_argument_identity_are_strict(env,tmp_path):
    child=spawn(env,'upload',('--channel','paper','--plan',str(env['path'])))
    try:
        time.sleep(.03)
        assert p.identify(child.pid,env,'paper') is None
    finally:cleanup(child)
    child=spawn(env,'controller')
    try:
        assert await_identity(child,env,'controller')
        env['path'].write_text('{"changed":true}')
        assert p.identify(child.pid,env,'controller') is None
    finally:cleanup(child)


def test_foreign_uid_and_unrelated_command_not_adopted(env,monkeypatch):
    child=spawn(env,'controller')
    try:
        assert await_identity(child,env,'controller')
        uid=os.getuid();monkeypatch.setattr(p.os,'getuid',lambda:uid+1)
        assert p.identify(child.pid,env,'controller') is None
    finally:cleanup(child)


def test_pid_reuse_between_double_checks_never_sends_signal(env):
    original=dict(pid=123,uid=os.getuid(),start_ticks='10',args=['original'],role='controller')
    calls=[]
    def identify(*args,**kwargs):
        calls.append(True)
        return original if len(calls)==1 else dict(original,start_ticks='11')
    sent=[]
    result=p.stop_owned(original,env,identify_fn=identify,send=lambda *args:sent.append(args))
    assert result['status']=='PROCESS_IDENTITY_CHANGED_NO_SIGNAL' and sent==[]


def test_owned_real_process_only_receives_graceful_sigterm(env):
    child=spawn(env,'controller')
    try:
        row=await_identity(child,env,'controller');assert row
        sent=[]
        def signal_one(pid,number):sent.append((pid,number));os.kill(pid,number)
        result=p.stop_owned(row,env,send=signal_one,grace=1)
        assert result['status']=='OWNED_HELPER_ENDED'
        assert sent==[(child.pid,signal.SIGTERM)]
        child.wait(timeout=3)
    finally:cleanup(child)


def test_lost_receipt_discovery_uses_exact_proc_metadata(env):
    child=spawn(env,'upload',('--channel','full'))
    try:
        assert await_identity(child,env,'full')
        rows=p.find_owned(env,'full',())
        assert any(row['pid']==child.pid for row in rows)
    finally:cleanup(child)


def test_only_exact_authorized_launch_plan_alias_is_accepted(env,tmp_path):
    alias=tmp_path/'preparation-plan.json';alias.write_bytes(env['path'].read_bytes())
    arbitrary=tmp_path/'same-bytes-but-not-authorized.json';arbitrary.write_bytes(alias.read_bytes())
    launch=dict(plan=str(alias),plan_sha256=env['header']['plan_sha256'],
                campaign_identity=env['header']['recipe_sha256'],python=sys.executable)
    (env['runtime']/'launch.json').write_text(json.dumps(launch))
    def child(plan,interval='120'):
        return subprocess.Popen([sys.executable,'-I','-S','-B','-u',
            str(env['package']/'paper3r11/platform/entry.py'),'upload','--channel','full',
            '--plan',str(plan),'--interval',interval],stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    valid=child(alias)
    try:assert await_identity(valid,env,'full')
    finally:cleanup(valid)
    invalid=child(arbitrary)
    try:
        time.sleep(.03);assert p.identify(invalid.pid,env,'full') is None
    finally:cleanup(invalid)
    launch['campaign_identity']='wrong';(env['runtime']/'launch.json').write_text(json.dumps(launch))
    invalid=child(alias)
    try:
        time.sleep(.03);assert p.identify(invalid.pid,env,'full') is None
    finally:cleanup(invalid)


@pytest.mark.parametrize('interval', ['nan','inf','0','-1','not-a-number'])
def test_invalid_native_interval_is_not_an_owned_management_command(env,interval):
    child=spawn(env,'upload',('--channel','paper','--interval',interval))
    try:
        time.sleep(.03);assert p.identify(child.pid,env,'paper') is None
    finally:cleanup(child)
