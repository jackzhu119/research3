import base64
import errno
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import subprocess
import sys
import zipfile

import pytest

from build_delivery import build

ROOT=Path(__file__).resolve().parents[1]


def load(path):
    spec=importlib.util.spec_from_file_location('r11_delivery_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def fixture(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'delivery_template.py').write_bytes((ROOT/'delivery_template.py').read_bytes())
    (source/'START.py').write_text('''import json,sys\nfrom pathlib import Path\np=Path(__file__).with_name('EXPLICIT_OPERATION.json')\np.write_text(json.dumps({'argv':sys.argv[1:],'fixture_only':True}))\n''')
    (source/'R10_MATH.md').write_text('Synthetic fixture, not scientific evidence.\n')
    record=build(source,files=['START.py','R10_MATH.md','delivery_template.py'])
    entry=source/'delivery/START-PAPER3-R10-OPTIMIZED.py'
    return source,record,entry,load(entry)


def test_default_actual_installer_is_offline_no_torch_jobs_or_dependency_install(tmp_path):
    source,record,entry,module=fixture(tmp_path);home=tmp_path/'home';home.mkdir()
    guard='''import importlib.abc,sys,runpy\nclass Guard(importlib.abc.MetaPathFinder):\n def find_spec(self,n,p=None,t=None):\n  if n.split('.')[0] in {'torch','numpy','requests','socket','subprocess'}:raise AssertionError('Forbidden install import '+n)\nsys.meta_path.insert(0,Guard())\nsys.argv=[sys.argv[1],'--home',sys.argv[2]]\nrunpy.run_path(sys.argv[0],run_name='__main__')\n'''
    completed=subprocess.run([sys.executable,'-I','-S','-B','-c',guard,str(entry),str(home)],capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr
    current=json.loads((home/'paper3-round10-optimized/CURRENT.json').read_text())
    package=Path(current['package'])
    assert not (package/'EXPLICIT_OPERATION.json').exists()
    assert current['training_jobs_submitted']==0 and not current['torch_imported']
    assert current['payload_sha256']==record['payload_sha256']


def test_repeated_install_preserves_code_mtime_and_all_prior_outputs(tmp_path):
    _,record,_,module=fixture(tmp_path);home=tmp_path/'home';home.mkdir()
    old=home/'projects/paper3/runs/followup_20261025_r9/checkpoint.pt';old.parent.mkdir(parents=True);old.write_bytes(b'original checkpoint')
    package=module.install(home)
    before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in package.rglob('*') if p.is_file()}
    again=module.install(home)
    assert package==again
    assert all(p.read_bytes()==data and p.stat().st_mtime_ns==stamp for p,(data,stamp) in before.items())
    assert old.read_bytes()==b'original checkpoint'


def test_modified_existing_immutable_code_refuses_overwrite(tmp_path):
    _,_,_,module=fixture(tmp_path);package=module.install(tmp_path/'home')
    (package/'START.py').write_text('user changed immutable code')
    with pytest.raises(ValueError,match='not overwritten'):module.install(tmp_path/'home')
    assert (package/'START.py').read_text()=='user changed immutable code'


def test_empty_owned_version_placeholder_is_recoverable(tmp_path):
    _,record,_,module=fixture(tmp_path);home=tmp_path/'home'
    target=home/'paper3-round10-optimized'/record['payload_sha256'][:20];target.mkdir(parents=True)
    assert module.install(home)==target
    assert (target/'START.py').is_file()


@pytest.mark.parametrize('bad_path',['../escape.py','/absolute.py','a/../escape.py','private/github.json','a\\b.py'])
def test_unsafe_zip_members_rejected_before_writing(tmp_path,bad_path):
    _,_,_,module=fixture(tmp_path);data=b'bad'
    manifest={'schema_version':'paper3-r10-optimized-code-package-v1','files':{bad_path:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}}}
    m=json.dumps(manifest).encode();buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as z:z.writestr('MANIFEST.json',m);z.writestr(bad_path,data)
    package=buffer.getvalue();module.PAYLOAD_B85=base64.b85encode(package).decode()
    module.PAYLOAD_SHA256=hashlib.sha256(package).hexdigest();module.MANIFEST_SHA256=hashlib.sha256(m).hexdigest()
    with pytest.raises(ValueError):module.install(tmp_path/'home')
    assert not (tmp_path/'escape.py').exists()


def test_zip_symlink_rejected_and_corrupt_payload_not_executed(tmp_path):
    _,_,_,module=fixture(tmp_path);data=b'/tmp/other'
    manifest={'schema_version':'paper3-r10-optimized-code-package-v1','files':{'link.py':{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}}}
    m=json.dumps(manifest).encode();buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as z:
        z.writestr('MANIFEST.json',m)
        info=zipfile.ZipInfo('link.py');info.create_system=3;info.external_attr=(stat.S_IFLNK|0o777)<<16;z.writestr(info,data)
    b=buffer.getvalue();module.PAYLOAD_B85=base64.b85encode(b).decode();module.PAYLOAD_SHA256=hashlib.sha256(b).hexdigest();module.MANIFEST_SHA256=hashlib.sha256(m).hexdigest()
    with pytest.raises(ValueError,match='links'):module.install(tmp_path/'home')
    module.PAYLOAD_SHA256='0'*64
    with pytest.raises(ValueError,match='package SHA256'):module.install(tmp_path/'home')


def test_explicit_cpu_validation_and_submission_are_distinct_routes(tmp_path):
    _,_,entry,module=fixture(tmp_path);home=tmp_path/'home'
    assert module.main(['--home',str(home),'--python',sys.executable,'--validate'])==0
    package=Path(json.loads((home/'paper3-round10-optimized/CURRENT.json').read_text())['package'])
    operation=json.loads((package/'EXPLICIT_OPERATION.json').read_text())
    assert operation['argv']==['validate','--cpu']
    assert module.main(['--home',str(home),'--python',sys.executable,'--submit','--plan','/explicit/r11-plan.json'])==0
    assert json.loads((package/'EXPLICIT_OPERATION.json').read_text())['argv']==['submit','--plan','/explicit/r11-plan.json','--submit']
    # START is a harmless argv-recording fixture; no Slurm or GPU is invoked.


def test_ambiguous_combined_modes_and_missing_plan_rejected(tmp_path):
    _,_,_,module=fixture(tmp_path)
    with pytest.raises(ValueError,match='one explicit'):module.main(['--home',str(tmp_path/'home'),'--prepare','mechanism','--submit'])
    with pytest.raises(ValueError,match='existing explicit --plan'):module.main(['--home',str(tmp_path/'home'),'--submit'])


def test_unsupported_directory_fsync_is_tolerated(tmp_path,monkeypatch):
    _,_,_,module=fixture(tmp_path);original=module.os.fsync
    def fsync(fd):
        if stat.S_ISDIR(module.os.fstat(fd).st_mode):raise OSError(errno.EINVAL,'fixture unsupported directory fsync')
        return original(fd)
    monkeypatch.setattr(module.os,'fsync',fsync)
    assert (module.install(tmp_path/'home')/'START.py').is_file()


def test_build_is_deterministic_and_code_only(tmp_path):
    source,first,_,_=fixture(tmp_path)
    (source/'private').mkdir();(source/'private/github.json').write_text('not-a-real-credential')
    (source/'evidence').mkdir();(source/'evidence/raw.npy').write_bytes(b'no raw data')
    second=build(source,files=['START.py','R10_MATH.md','delivery_template.py'])
    assert first==second
    manifest=json.loads((source/'delivery/MANIFEST.json').read_text())
    assert 'private/github.json' not in manifest['files'] and 'evidence/raw.npy' not in manifest['files']


def test_source_collection_includes_bounded_research_and_only_toplevel_evidence(tmp_path):
    from build_delivery import source_files
    source,_,_,_=fixture(tmp_path)
    research=source/'research';research.mkdir();(research/'audit.json').write_text('{"fixture":true}')
    (research/'evidence').mkdir();(research/'evidence/effects.csv').write_text('method,effect\nsoftware_fixture,0\n')
    (research/'notes.csv').write_text('unrequested,csv\n')
    early=source/'evidence/early';early.mkdir(parents=True);(early/'participants.csv').write_text('id,ba\nfixture,.5\n')
    raw=early/'restored';raw.mkdir();(raw/'private-output.json').write_text('{}')
    (early/'large-tree.json').write_bytes(b'x'*(4*1024*1024+1))
    (early/'signal.npy').write_bytes(b'never included')
    (early/'link.json').symlink_to(research/'audit.json')
    chosen={p.relative_to(source).as_posix() for p in source_files(source)}
    assert 'research/audit.json' in chosen and 'evidence/early/participants.csv' in chosen
    assert 'research/evidence/effects.csv' in chosen and 'research/notes.csv' not in chosen
    assert not {'evidence/early/restored/private-output.json','evidence/early/large-tree.json',
                'evidence/early/signal.npy','evidence/early/link.json'}&chosen


def test_all_in_one_start_prepares_by_default_and_requires_explicit_submit(tmp_path,monkeypatch):
    source,_,_,module=fixture(tmp_path);calls=[]
    monkeypatch.setattr(module,'install',lambda _:source)
    monkeypatch.setattr(subprocess,'run',lambda command,**kw: calls.append(command) or type('Done',(),{'returncode':0})())
    module.main(['--home',str(tmp_path/'home'),'--python',sys.executable,'--continue'])
    assert calls[-1][-3:]==['continue','--root','/public/home/zhuziyuan/projects/paper3']
    assert '--submit' not in calls[-1]
    module.main(['--home',str(tmp_path/'home'),'--python',sys.executable,'--continue','--submit'])
    assert 'continue' in calls[-1] and calls[-1][-1]=='--submit'
