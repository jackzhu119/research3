"""Isolated standalone installer boundaries; no network, Slurm or GPU calls."""
from __future__ import annotations
import base64
import errno
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load(ROOT/'build_delivery.py', 'r11_management_test_builder')


@pytest.fixture
def artifact(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    for name in builder.REQUIRED:
        (source/name).write_text('VALUE = 1\n')
    for name in ('build_delivery.py', 'installer_template.py', 'entry.py'):
        shutil.copy2(ROOT/name, source/name)
    (source/'supervisor.py').write_text('''import json,os,sys
from pathlib import Path
def main(argv=None):
 a=list(sys.argv[1:] if argv is None else argv)
 p=Path(os.environ['R11_INSTALL_TEST_ARGS']);p.write_text(json.dumps(a))
 assert not any(k.split('.')[0] in {'torch','numpy','scipy','sklearn'} for k in sys.modules)
 return int(os.environ.get('R11_INSTALL_TEST_RETURN','0'))
''')
    (source/'README.md').write_text('A management-only repair fixture.\n')
    tests=source/'tests';tests.mkdir();(tests/'test_fixture.py').write_text('assert True\n')
    proof=source/'proof';proof.mkdir();(proof/'tiny.json').write_text('{"school_calls":0}\n')
    record=builder.build(source)
    entry=source/'delivery'/record['entry']
    module=load(entry, 'r11_repair_fixture_'+tmp_path.name)
    home=tmp_path/'home';home.mkdir()
    project=tmp_path/'project with spaces';project.mkdir()
    plan=project/'runs/r11_campaign/plan.json';plan.parent.mkdir(parents=True)
    plan.write_text('{"existing_recipe":"unchanged"}')
    return dict(source=source,home=home,project=project,plan=plan,
                record=record,entry=entry,module=module,audit=tmp_path/'dispatch.json')


def invoke(value,*args,env_extra=None):
    env=dict(os.environ,R11_INSTALL_TEST_ARGS=str(value['audit']),PYTHONDONTWRITEBYTECODE='1')
    env.update(env_extra or {})
    return subprocess.run([sys.executable,'-I','-S','-B',str(value['entry']),
        '--home',str(value['home']),'--plan',str(value['plan']),'--python',sys.executable,*args],
        capture_output=True,text=True,timeout=15,env=env)


def snapshot(root):
    return {p.relative_to(root).as_posix():(p.read_bytes(),p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def test_actual_default_installs_twice_without_dispatch_or_science(artifact):
    x=artifact; before=snapshot(x['project'])
    first=invoke(x);assert first.returncode==0,first.stderr
    assert 'R11_MANAGEMENT_CODE_READY_NO_ACTION' in first.stdout
    assert not x['audit'].exists() and snapshot(x['project'])==before
    folder=x['module'].package_path(x['home']);installed=snapshot(folder)
    second=invoke(x);assert second.returncode==0,second.stderr
    assert snapshot(folder)==installed and not x['audit'].exists()
    assert x['module'].verify_install(folder,*x['module'].checked_payload()[1:])


def test_default_import_is_standard_library_only(artifact):
    script="import runpy,sys;m=runpy.run_path(sys.argv[1],run_name='installer_only');assert not any(k.split('.')[0] in {'torch','numpy','scipy','sklearn'} for k in sys.modules);print(m['SCHEMA'])"
    result=subprocess.run([sys.executable,'-I','-S','-B','-c',script,str(artifact['entry'])],
                          capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='paper3-r11-management-recovery-package-v1'


def test_explicit_apply_dispatches_exact_existing_plan_with_space_paths(artifact):
    x=artifact; package=x['project']/'old scientific package';work=x['project']/'recovery work'
    credentials=x['project']/'private token file.json'
    before=snapshot(x['project'])
    result=invoke(x,'--apply','--submit','--package',str(package),'--work',str(work),
                  '--credentials',str(credentials))
    assert result.returncode==0,result.stderr
    assert json.loads(x['audit'].read_text())==['start','--plan',str(x['plan']),'--submit',
        '--package',str(package),'--work',str(work),'--credentials',str(credentials),'--python',sys.executable]
    assert snapshot(x['project'])==before


def test_management_failure_status_is_returned_without_second_dispatch(artifact):
    x=artifact;result=invoke(x,'--apply','--submit',env_extra={'R11_INSTALL_TEST_RETURN':'17'})
    assert result.returncode==17
    assert json.loads(x['audit'].read_text())[0]=='start'


def test_check_requires_exact_existing_install_and_never_installs(artifact):
    x=artifact;result=invoke(x,'--check')
    assert result.returncode!=0
    assert not (x['home']/'paper3-r11-management-recovery').exists()
    assert not x['audit'].exists()
    assert invoke(x).returncode==0
    before=snapshot(x['home'])
    checked=invoke(x,'--check')
    assert checked.returncode==0,checked.stderr
    assert json.loads(x['audit'].read_text())==['check','--plan',str(x['plan']),'--python',sys.executable]
    assert snapshot(x['home'])==before


@pytest.mark.parametrize('flags',[('--submit',),('--apply',),('--check','--submit'),
    ('--check','--apply','--submit'),('--check','--install')])
def test_incomplete_or_conflicting_activation_fails_before_install(artifact,flags):
    x=artifact;result=invoke(x,*flags)
    assert result.returncode!=0
    assert not (x['home']/'paper3-r11-management-recovery').exists()
    assert not x['audit'].exists()


def test_builder_is_deterministic_and_excludes_private_data(artifact):
    x=artifact;first=x['entry'].read_bytes()
    for relative in ('private/github.json','tests/private/github.json','proof/credentials/secret.json'):
        p=x['source']/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{"token":"fake"}')
    (x['source']/'checkpoint.pt').write_bytes(b'not code')
    (x['source']/'github.json').write_text('{"token":"fake"}')
    second=builder.build(x['source'])
    assert second==x['record'] and first==x['entry'].read_bytes()
    with zipfile.ZipFile(x['source']/'delivery/paper3-r11-management-recovery.zip') as z:
        assert 'proof/tiny.json' in z.namelist()
        assert not any('private' in name or 'credential' in name or name.endswith('.pt') for name in z.namelist())


def test_existing_tampered_immutable_package_never_overwritten(artifact):
    x=artifact;m=x['module'];folder=m.install(x['home']);target=folder/'entry.py'
    target.write_text('changed\n')
    with pytest.raises(ValueError,match='EXISTING_IMMUTABLE_PACKAGE_DIFFERS'):m.install(x['home'])
    assert target.read_text()=='changed\n' and not x['audit'].exists()


def test_extra_unmanifested_python_cannot_inject_imports(artifact):
    x=artifact;m=x['module'];folder=m.install(x['home'])
    (folder/'ssl.py').write_text('raise RuntimeError("injected module")\n')
    with pytest.raises(ValueError,match='EXISTING_IMMUTABLE_PACKAGE_DIFFERS'):m.install(x['home'])


def test_nested_symlink_cannot_satisfy_manifest(artifact,tmp_path):
    x=artifact;m=x['module'];folder=m.install(x['home']);tests=folder/'tests';shutil.rmtree(tests)
    outside=tmp_path/'outside';outside.mkdir();(outside/'test_fixture.py').write_text('assert True\n')
    tests.symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='EXISTING_IMMUTABLE_PACKAGE_DIFFERS'):m.install(x['home'])


def forged(module,members,*,descriptors=None,symlink=None):
    values=dict(members)
    inventory=descriptors or {name:dict(bytes=len(data),sha256=module.sha(data)) for name,data in values.items()}
    metadata=module.encoded(dict(schema_version=module.SCHEMA,files=inventory))
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as z:
        z.writestr('MANIFEST.json',metadata)
        for name,data in values.items():
            info=zipfile.ZipInfo(name);info.create_system=3
            info.external_attr=((stat.S_IFLNK if name==symlink else stat.S_IFREG)|0o600)<<16
            z.writestr(info,data)
    payload=buffer.getvalue()
    return dict(PAYLOAD_B85=base64.b85encode(payload).decode(),PAYLOAD_SHA256=module.sha(payload),
                MANIFEST_SHA256=module.sha(metadata))


@pytest.mark.parametrize('name',['../escape.py','/absolute.py','tests/../escape.py',
    'private/github.json','tests\\escape.py'])
def test_unsafe_archive_paths_are_rejected_before_any_install(artifact,monkeypatch,name):
    x=artifact;m=x['module']
    for key,value in forged(m,{name:b'bad','entry.py':b'pass\n','supervisor.py':b'pass\n'}).items():
        monkeypatch.setattr(m,key,value)
    with pytest.raises(ValueError,match='UNSAFE_PACKAGE_MEMBER_PATH'):m.install(x['home'])
    assert not (x['home']/'paper3-r11-management-recovery').exists()


def test_symlink_archive_member_rejected_before_publish(artifact,monkeypatch):
    x=artifact;m=x['module'];values={'entry.py':b'pass\n','supervisor.py':b'pass\n'}
    for key,value in forged(m,values,symlink='entry.py').items():monkeypatch.setattr(m,key,value)
    with pytest.raises(ValueError,match='PACKAGE_MEMBER_TYPE_OR_SIZE_DIFFERS'):m.install(x['home'])
    assert not (x['home']/'paper3-r11-management-recovery').exists()


def test_failed_member_hash_cannot_publish_partial_install(artifact,monkeypatch):
    x=artifact;m=x['module'];values={'entry.py':b'pass\n','supervisor.py':b'pass\n'}
    descriptors={name:dict(bytes=len(data),sha256='0'*64) for name,data in values.items()}
    for key,value in forged(m,values,descriptors=descriptors).items():monkeypatch.setattr(m,key,value)
    with pytest.raises(ValueError,match='PACKAGE_MEMBER_SHA256_DIFFERS'):m.install(x['home'])
    assert not m.package_path(x['home']).exists()
    assert not list((x['home']/m.BASE_NAME).glob('.install-*'))


def test_payload_hash_failure_prevents_install_or_dispatch(artifact,monkeypatch):
    x=artifact;m=x['module'];monkeypatch.setattr(m,'PAYLOAD_SHA256','0'*64)
    with pytest.raises(ValueError,match='PACKAGE_SHA256_DIFFERS'):m.install(x['home'])
    assert not (x['home']/m.BASE_NAME).exists() and not x['audit'].exists()


def test_installer_receipt_retries_lost_staging_without_replacing_science(artifact,monkeypatch):
    x=artifact;m=x['module'];original=m.os.replace;attempts=[]
    def replace(source,target):
        if Path(target).name=='CURRENT.json':
            attempts.append(Path(source))
            if len(attempts)==1:
                Path(source).unlink()
                raise FileNotFoundError(errno.ENOENT,'injected staging loss')
        return original(source,target)
    monkeypatch.setattr(m.os,'replace',replace)
    folder=m.install(x['home'])
    assert len(attempts)==2 and len(set(attempts))==2
    assert m.verify_install(folder,*m.checked_payload()[1:])
    assert not x['audit'].exists()


def test_installer_receipt_accepts_exact_ambiguous_replace_ack(artifact,monkeypatch):
    x=artifact;m=x['module'];original=m.os.replace
    def replace(source,target):
        original(source,target)
        if Path(target).name=='CURRENT.json':raise OSError(errno.EIO,'injected lost ack')
    monkeypatch.setattr(m.os,'replace',replace)
    folder=m.install(x['home'])
    assert m.verify_install(folder,*m.checked_payload()[1:])


def test_mutable_r11_status_may_advance_while_installer_dispatches(artifact,monkeypatch):
    x=artifact;m=x['module'];state=x['plan'].parent/'runtime/status.json'
    state.parent.mkdir();state.write_text('{"complete_tasks":900}')
    scientific=x['project']/'p3work/spatial_geometry.py';scientific.parent.mkdir()
    scientific.write_bytes(b'immutable old scientific source')
    result=x['plan'].parent/'completed/result.json';result.parent.mkdir();result.write_bytes(b'immutable old result')
    calls=[]
    def run(command,**kwargs):
        calls.append(command);state.write_text('{"complete_tasks":901}')
        return type('Reply',(),{'returncode':0})()
    monkeypatch.setattr(subprocess,'run',run)
    assert m.main(['--home',str(x['home']),'--plan',str(x['plan']),'--python',sys.executable,
                   '--apply','--submit'])==0
    assert len(calls)==1 and json.loads(state.read_text())['complete_tasks']==901
    assert scientific.read_bytes()==b'immutable old scientific source'
    assert result.read_bytes()==b'immutable old result'
