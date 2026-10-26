"""Verify immutable code-only installation and isolated argv forwarding."""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zlib

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('completed_upload_builder',HERE/'build.py')
BUILDER=importlib.util.module_from_spec(spec);spec.loader.exec_module(BUILDER)


def fixture(tmp_path,driver=None):
    source=tmp_path/'public-source'
    for name in BUILDER.REQUIRED_FILES:
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('# public code only: '+name+'\n')
    (source/'upload_driver.py').write_text(driver or '''import json,os,pathlib,sys
def main(argv=None):
 proof={"argv":list(argv or []),"torch_imported":"torch" in sys.modules,"isolated":sys.flags.isolated,"no_site":sys.flags.no_site}
 pathlib.Path(os.environ["UPLOAD_DELIVERY_PROOF"]).write_text(json.dumps(proof))
 print("STUB_UPLOAD_BACKGROUND_STARTED",flush=True)
 return 0
''')
    (source/'private').mkdir();(source/'private/github.json').write_text('{"token":"never package this"}')
    (source/'work').mkdir();(source/'work/predictions.bin').write_bytes(b'never package output')
    output=tmp_path/'delivery/START-PAPER3-COMPLETED-UPLOAD.py'
    metadata=BUILDER.build(source,output)
    spec=importlib.util.spec_from_file_location('completed_upload_entry_'+metadata['payload_sha256'][:12],output)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return source,output,metadata,module


def test_build_is_deterministic_and_explicitly_excludes_work_private_tests(tmp_path):
    source,output,first,module=fixture(tmp_path)
    before=output.read_bytes();second=BUILDER.build(source,output)
    assert first==second and output.read_bytes()==before
    assert set(module.unpack_payload())==set(BUILDER.REQUIRED_FILES)
    assert not any('private' in name or 'test_' in name or 'work/' in name for name in module.FILE_HASHES)
    assert b'never package this' not in b''.join(module.unpack_payload().values())


def test_install_is_idempotent_and_preserves_owned_source_bytes_and_mtimes(tmp_path):
    source,output,metadata,module=fixture(tmp_path)
    base=tmp_path/'home/paper3-completed-results-uploader'
    installed=module.install(base)
    before={name:((installed/name).read_bytes(),(installed/name).stat().st_mtime_ns) for name in module.FILE_HASHES}
    assert installed.name==metadata['payload_sha256'][:20]
    assert module.install(base)==installed
    assert all((installed/name).read_bytes()==data and (installed/name).stat().st_mtime_ns==mtime
               for name,(data,mtime) in before.items())
    assert (installed/'INSTALL_MANIFEST.json').is_file()
    assert not list(base.glob('*.tmp-*'))


@pytest.mark.parametrize('kind',['packed_sha','source_sha','source_path','driver_absent'])
def test_corrupt_payload_fails_before_any_install_write(tmp_path,kind):
    source,output,metadata,module=fixture(tmp_path)
    base=tmp_path/'must-not-exist'
    if kind=='packed_sha':module.PAYLOAD_SHA256='0'*64
    elif kind=='source_sha':module.FILE_HASHES['upload_driver.py']='0'*64
    else:
        files=module.unpack_payload()
        if kind=='source_path':files['../outside.py']=b'raise AssertionError("not installed")\n'
        else:del files['upload_driver.py']
        document={name:data.decode('utf-8') for name,data in files.items()}
        packed=zlib.compress(json.dumps(document).encode())
        module.PAYLOAD_SHA256=hashlib.sha256(packed).hexdigest()
        module.PAYLOAD_B85=base64.b85encode(packed).decode()
        module.FILE_HASHES={name:hashlib.sha256(data).hexdigest() for name,data in files.items()}
    with pytest.raises(ValueError):module.install(base)
    assert not base.exists()
    assert not (tmp_path/'outside.py').exists()


def test_corrupted_existing_package_is_not_silently_overwritten(tmp_path):
    source,output,metadata,module=fixture(tmp_path)
    base=tmp_path/'install';installed=module.install(base)
    path=installed/'collector.py';path.write_bytes(b'locally changed code\n')
    with pytest.raises(RuntimeError,match='source differs'):
        module.install(base)
    assert path.read_bytes()==b'locally changed code\n'


@pytest.mark.parametrize('argv',[
    ['--start','--root','/existing/project','--work-dir','/existing/work','--rounds','4','5','6','8'],
    ['--worker','--channel','paper','--root','/existing/project','--rounds','4','8'],
    ['--worker','--channel','full','--root','/existing/project','--credentials','/missing/credentials.json'],
])
def test_real_isolated_bootstrap_forwards_exact_driver_args_without_credential_gate(tmp_path,argv):
    source,output,metadata,module=fixture(tmp_path)
    proof=tmp_path/'proof.json';base=tmp_path/'installed'
    result=subprocess.run([sys.executable,'-I','-S','-B','-u',str(output),'--install-base',str(base),*argv],
        env={**os.environ,'UPLOAD_DELIVERY_PROOF':str(proof)},capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
    value=json.loads(proof.read_text())
    assert value['argv']==argv
    assert value['isolated']==value['no_site']==1 and value['torch_imported'] is False
    assert 'COMPLETED_UPLOAD_PACKAGE_READY' in result.stdout
    assert 'STUB_UPLOAD_BACKGROUND_STARTED' in result.stdout


def test_default_invocation_selects_background_start(tmp_path):
    source,output,metadata,module=fixture(tmp_path)
    proof=tmp_path/'proof.json'
    result=subprocess.run([sys.executable,'-I','-S','-B','-u',str(output),'--install-base',str(tmp_path/'installed')],
        env={**os.environ,'UPLOAD_DELIVERY_PROOF':str(proof)},capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(proof.read_text())['argv']==['--start']


def test_install_only_does_not_import_driver_dependencies_or_walk_user_sources(tmp_path):
    source,output,metadata,module=fixture(tmp_path,driver='raise AssertionError("Driver imported during install")\n')
    base=tmp_path/'installed';forbidden_root=tmp_path/'project'
    forbidden_root.mkdir();(forbidden_root/'large-data.bin').write_bytes(b'never read during code install')
    code='''import pathlib,runpy,sys
class Forbidden:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split(".")[0] in ("torch","numpy","subprocess","requests","socket") or fullname in ("urllib.request","http.client"):
   raise AssertionError("No compute, job or network module during install: "+fullname)
def no_walk(*args,**kwargs):raise AssertionError("No source directory scan during install")
pathlib.Path.glob=no_walk;pathlib.Path.rglob=no_walk;pathlib.Path.iterdir=no_walk
sys.meta_path.insert(0,Forbidden())
sys.argv=[sys.argv[1],"--install-base",sys.argv[2],"--install-only","--root",sys.argv[3]]
runpy.run_path(sys.argv[0],run_name="__main__")
'''
    result=subprocess.run([sys.executable,'-I','-S','-B','-u','-c',code,str(output),str(base),str(forbidden_root)],
        capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'COMPLETED_UPLOAD_PACKAGE_READY' in result.stdout
    assert (forbidden_root/'large-data.bin').read_bytes()==b'never read during code install'


def test_new_source_payload_installs_side_by_side_without_touching_old_package(tmp_path):
    source,output,metadata,module=fixture(tmp_path)
    base=tmp_path/'installed';old=module.install(base);before={p:p.read_bytes() for p in old.rglob('*') if p.is_file()}
    (source/'collector.py').write_text('# improved public collector\n')
    new_output=tmp_path/'next/START.py';new_meta=BUILDER.build(source,new_output)
    spec=importlib.util.spec_from_file_location('new_version',new_output);new=importlib.util.module_from_spec(spec);spec.loader.exec_module(new)
    installed=new.install(base)
    assert installed!=old and installed.name==new_meta['payload_sha256'][:20]
    assert all(path.read_bytes()==data for path,data in before.items())


def test_missing_required_builder_input_and_credential_literal_are_rejected(tmp_path):
    source,output,metadata,module=fixture(tmp_path)
    missing=source/'paper_credentials.py';missing.unlink()
    with pytest.raises(FileNotFoundError,match='not ready'):
        BUILDER.build(source,output)
    missing.write_text('TOKEN="github_pat_"+"x"\n')
    (source/'upload_driver.py').write_text('TOKEN="github_pat_'+'x'*30+'"\n')
    with pytest.raises(ValueError,match='Credential-shaped'):
        BUILDER.build(source,output)
