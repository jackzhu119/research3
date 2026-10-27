from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import install_hooks as hooks


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def helpers(tmp_path):
    target = tmp_path/'helper-package'; target.mkdir()
    sources = {
        'verified_context_cache.py': '''from contextlib import contextmanager
from pathlib import Path
@contextmanager
def active(root, action):
 p=Path(root)/'events.txt'
 with p.open('a') as h:h.write('cache-enter\\n')
 try:yield
 finally:
  with p.open('a') as h:h.write('cache-exit\\n')
''',
        'newdiagnostics.py': '''from contextlib import contextmanager
def install(module):
 old=module.active
 @contextmanager
 def active(*a,**k):
  k['traceback_interval']=0
  with old(*a,**k):yield
 module.active=active
'''}
    rows = {}
    for name, text in sources.items():
        path = target/name; path.write_text(text)
        rows[name] = dict(bytes=len(path.read_bytes()), sha256=hooks.sha(path.read_bytes()))
    (target/'MANIFEST.json').write_text(json.dumps(dict(files=rows)))
    return target


STARTUP = '''from contextlib import contextmanager
from pathlib import Path
@contextmanager
def active(root, action, *, modules=None, torch_module=None, emit=None,
           action_file=None, traceback_interval=300):
 with (Path(root)/'events.txt').open('a') as h:h.write('original:'+str(traceback_interval)+'\\n')
 try:yield
 finally:
  with (Path(root)/'events.txt').open('a') as h:h.write('original-exit\\n')
# PAPER3_R1_PERSISTENT_DIGEST_CACHE_V1
# prior cache wrapper retained byte for byte
# END_PAPER3_R1_PERSISTENT_DIGEST_CACHE_V1
# PAPER3_R1_RESUME_ACTION_V3
# prior clock/driver wrapper retained byte for byte
# END_PAPER3_R1_RESUME_ACTION_V3
'''


def project(tmp_path):
    root = tmp_path/'project'; home = tmp_path/'home'; native = home/'paper3-r1-repairs/test/active-round1/test'
    native.mkdir(parents=True); root.mkdir()
    (native/'action_startup.py').write_text(STARTUP)
    for name in ('worker.py','kernel_bridge.py','audited_runtime.py','START.py','scheduler.py'):
        (native/name).write_text('# original '+name+'\n')
    runtime = root/hooks.RUNTIME; runtime.mkdir(parents=True)
    (runtime/'launch.json').write_text(json.dumps(dict(package=str(native))))
    (runtime/'scheduler-state.json').write_text(json.dumps(dict(marker='paper3r1p-0123456789abcdef', slots={})))
    return root, home, native


def test_generated_hook_real_context_composition_and_prior_blocks_preserved(tmp_path):
    repair = helpers(tmp_path)
    fixed = hooks.patch_source(STARTUP.encode(), hooks.checked_helpers(repair))
    assert hooks.strip_own_block(fixed.decode()) == STARTUP
    path = tmp_path/'action_startup.py'; path.write_bytes(fixed)
    startup = module(path, 'fixture_r1_action_startup')
    with startup.active(tmp_path, dict(key='B4'), traceback_interval=300):
        with (tmp_path/'events.txt').open('a') as h:h.write('body\n')
    assert (tmp_path/'events.txt').read_text().splitlines() == [
        'cache-enter', 'original:0', 'body', 'original-exit', 'cache-exit']


def test_action_exception_still_restores_both_contexts(tmp_path):
    repair = helpers(tmp_path)
    path = tmp_path/'startup.py'
    path.write_bytes(hooks.patch_source(STARTUP.encode(), hooks.checked_helpers(repair)))
    startup = module(path, 'fixture_r1_raising_startup')
    with pytest.raises(RuntimeError, match='original failure'):
        with startup.active(tmp_path, dict(key='B5')):
            raise RuntimeError('original failure')
    assert (tmp_path/'events.txt').read_text().splitlines()[-2:] == ['original-exit', 'cache-exit']


def test_activation_marker_bound_to_exact_owned_attempt(tmp_path):
    repair = helpers(tmp_path); rows = hooks.checked_helpers(repair)
    path = tmp_path/'startup.py'; path.write_bytes(hooks.patch_source(STARTUP.encode(), rows))
    startup = module(path, 'fixture_r1_bound_activation')
    action = dict(key='development-test'); token = 'abc123'
    attempt = tmp_path/hooks.RUNTIME/'actions'/hooks.sha(action['key'].encode())/'attempts'/token
    attempt.mkdir(parents=True); action_file = attempt/'action.json'
    action_file.write_text(json.dumps(dict(action=action, claim=dict(token=token))))
    with startup.active(tmp_path, action, action_file=action_file):
        active = json.loads((attempt/'repair-active.json').read_text())
        assert active['repair_id'] == 'paper3-r1-complete-runtime-v1'
        assert active['action_key'] == action['key'] and active['claim_token'] == token
        assert active['helpers_sha256'] == {n:digest for n,(_,digest) in rows.items()}


def test_activation_marker_not_written_for_unbound_action_file(tmp_path):
    repair = helpers(tmp_path)
    path = tmp_path/'startup.py'
    path.write_bytes(hooks.patch_source(STARTUP.encode(), hooks.checked_helpers(repair)))
    startup = module(path, 'fixture_r1_unbound_activation')
    action = dict(key='development-test'); action_file = tmp_path/'action.json'
    action_file.write_text(json.dumps(dict(action=action, claim=dict(token='abc123'))))
    with startup.active(tmp_path, action, action_file=action_file):
        pass
    assert not (tmp_path/'repair-active.json').exists()


def test_apply_idempotent_only_operational_file_and_immutable_helpers(tmp_path):
    root, home, native = project(tmp_path); repair = helpers(tmp_path)
    before = {p:p.read_bytes() for p in native.iterdir()}
    metadata = {p:p.read_bytes() for p in (root/hooks.RUNTIME).glob('*.json')}
    first = hooks.apply(root, home, package=repair)
    assert first['scientific_files_changed'] == first['jobs_submitted'] == first['signals_sent'] == 0
    expected = (native/'action_startup.py').read_bytes(); modified = (native/'action_startup.py').stat().st_mtime_ns
    again = hooks.apply(root, home, package=repair)
    assert (native/'action_startup.py').read_bytes() == expected
    assert (native/'action_startup.py').stat().st_mtime_ns == modified
    assert again['operational_files'][0]['changed'] is False
    assert hooks.strip_own_block(expected.decode()).encode() == before[native/'action_startup.py']
    for path, data in before.items():
        if path.name != 'action_startup.py':assert path.read_bytes() == data
    for path, data in metadata.items():assert path.read_bytes() == data
    assert hooks.inspect(root, home)['startup_files'][0]['installed']


def test_apply_rejects_tampered_helper_before_switching(tmp_path):
    root, home, native = project(tmp_path); repair = helpers(tmp_path)
    (repair/'verified_context_cache.py').write_text('tampered')
    with pytest.raises(ValueError, match='REPAIR_HELPER_SHA256_DIFFERS'):
        hooks.apply(root, home, package=repair)
    assert (native/'action_startup.py').read_text() == STARTUP


def test_apply_refuses_startup_symlink(tmp_path):
    root, home, native = project(tmp_path); repair = helpers(tmp_path)
    target = native/'action_startup.py'; other = tmp_path/'real.py'; target.rename(other); target.symlink_to(other)
    with pytest.raises(ValueError, match='R1_STARTUP_NOT_OWNED_REGULAR_FILE'):
        hooks.apply(root, home, package=repair)
    assert other.read_text() == STARTUP


def test_partial_operational_apply_rolls_back_before_error(tmp_path, monkeypatch):
    root, home, first = project(tmp_path); repair = helpers(tmp_path)
    second = home/'paper3-r1-repairs/test2/active-round1/test'
    shutil.copytree(first, second)
    (root/hooks.RUNTIME/'scheduler-state.json').write_text(json.dumps(dict(slots={
        '1':dict(attempts=[dict(command=['python','worker.py','--package',str(second)])])})))
    before = {p:p.read_bytes() for folder in (first,second) for p in folder.iterdir()}
    original = hooks.atomic
    def fail_second(path, data):
        if Path(path) == second/'action_startup.py' and hooks.MARKER.encode() in data:
            raise OSError('synthetic shared filesystem failure')
        return original(path,data)
    monkeypatch.setattr(hooks,'atomic',fail_second)
    with pytest.raises(OSError,match='synthetic shared filesystem failure'):
        hooks.apply(root,home,package=repair)
    for path, data in before.items():
        assert path.read_bytes()==data


def test_own_block_replacement_does_not_accumulate_hooks(tmp_path):
    repair = helpers(tmp_path); helper_rows = hooks.checked_helpers(repair)
    once = hooks.patch_source(STARTUP.encode(), helper_rows)
    assert hooks.patch_source(once, helper_rows) == once
    with pytest.raises(ValueError, match='INCOMPLETE_OR_DUPLICATE'):
        hooks.patch_source(once + ('\n# '+hooks.MARKER+'\n').encode(), helper_rows)


@pytest.fixture
def delivery(tmp_path):
    source = tmp_path/'build'; source.mkdir()
    for p in ROOT.glob('*.py'):
        shutil.copy2(p, source/p.name)
    if (ROOT/'vendor').is_dir():
        shutil.copytree(ROOT/'vendor', source/'vendor',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # A dispatch stub is sufficient for installer-only tests. The real runtime
    # modules have their own independent production-module tests.
    for name in ('atomic_io.py','supervisor.py','parallel_runtime.py','newnode_tool.py'):
        p = source/name
        if not p.exists():p.write_text('# fixture placeholder; not executed by install-only tests\n')
    builder = module(source/'build_delivery.py', '_fixture_r1_delivery_builder')
    record = builder.build(source)
    return source/'delivery'/record['entry'], record


def test_actual_offline_default_install_no_torch_network_or_subprocess(delivery, tmp_path):
    entry, record = delivery; home = tmp_path/'home'
    code = '''import runpy,sys
def audit(event,args):
 if event in ('subprocess.Popen','socket.__new__','socket.connect'):raise AssertionError(event)
 if event=='import' and args[0].split('.')[0] in ('torch','numpy','requests'):raise AssertionError(args[0])
sys.addaudithook(audit)
sys.argv=[sys.argv[1],'--home',sys.argv[2]]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    command = [sys.executable,'-I','-S','-B','-c',code,str(entry),str(home)]
    first = subprocess.run(command,capture_output=True,text=True,timeout=15)
    assert first.returncode == 0, first.stderr
    assert 'R1_COMPLETE_REPAIR_CODE_READY_NO_ACTION' in first.stdout
    installed = home/'paper3-r1-full-repair'/record['payload_sha256'][:20]
    manifest = json.loads((installed/'MANIFEST.json').read_text())
    for name,row in manifest['files'].items():
        data=(installed/name).read_bytes()
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    stamps={p:p.stat().st_mtime_ns for p in installed.rglob('*') if p.is_file()}
    again = subprocess.run(command,capture_output=True,text=True,timeout=15)
    assert again.returncode == 0, again.stderr
    assert stamps=={p:p.stat().st_mtime_ns for p in installed.rglob('*') if p.is_file()}


@pytest.mark.parametrize('args', [['--submit'],['--repair'],['--refresh-pretraining']])
def test_installation_cannot_accidentally_activate(delivery, tmp_path, args):
    entry, _ = delivery; home=tmp_path/'home'
    result=subprocess.run([sys.executable,'-I','-S','-B',str(entry),'--home',str(home),*args],
        capture_output=True,text=True,timeout=10)
    assert result.returncode != 0
    assert not home.exists()


def test_explicit_dispatch_forwards_root_home_and_installed_package(delivery, tmp_path, monkeypatch):
    entry, record = delivery; installer=module(entry,'_fixture_r1_installer_dispatch');calls=[]
    monkeypatch.setattr(subprocess,'run',lambda command,**kw:calls.append(command) or type('R',(),dict(returncode=0))())
    home=tmp_path/'home';root=tmp_path/'project'
    assert installer.main(['--home',str(home),'--root',str(root),'--repair','--submit','--python',sys.executable])==0
    command=calls[0]
    assert Path(command[5]).name=='supervisor.py' and command[6]=='start'
    assert command[command.index('--root')+1]==str(root)
    assert command[command.index('--home')+1]==str(home)
    assert command[command.index('--package')+1]==str(home/'paper3-r1-full-repair'/record['payload_sha256'][:20])
    assert command.count('--submit')==1
    assert 'sbatch' not in command and 'scancel' not in command
