import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def toy_master(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('_master_build',HERE/'build_master.py')
    build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
    a=tmp_path/'r1.py';b=tmp_path/'upload.py'
    a.write_text('import sys;print("R1_COMPONENT_MODE="+sys.argv[1],flush=True);raise SystemExit(3 if sys.argv[1]=="--apply" else 0)\n')
    b.write_text('import sys;print("UPLOAD_COMPONENT_MODE="+sys.argv[1],flush=True)\n')
    monkeypatch.setattr(build,'ROOT',tmp_path)
    monkeypatch.setattr(build,'SOURCES',{'START-PAPER3-R1-1GPU.py':a,'START-PAPER3-MISSING-RESULTS.py':b})
    build.build()
    return tmp_path/'START-PAPER3-R1-AND-UPLOAD.py'


def test_one_component_error_does_not_prevent_independent_upload_start(tmp_path,monkeypatch):
    entry=toy_master(tmp_path,monkeypatch)
    result=subprocess.run([sys.executable,'-I','-S','-B',str(entry),'--apply'],capture_output=True,text=True,timeout=10)
    assert result.returncode==1 and 'UPLOAD_COMPONENT_MODE=--start' in result.stdout
    final=json.loads(result.stdout.splitlines()[-1])
    assert final['components']==[dict(component='R1',exit_code=3),dict(component='RESULTS',exit_code=0)]


def test_check_never_invokes_apply_or_start(tmp_path,monkeypatch):
    entry=toy_master(tmp_path,monkeypatch)
    result=subprocess.run([sys.executable,'-I','-S','-B',str(entry),'--check'],capture_output=True,text=True,timeout=10)
    assert result.returncode==0 and 'R1_COMPONENT_MODE=--check' in result.stdout
    assert 'UPLOAD_COMPONENT_MODE=--check' in result.stdout
    assert 'COMPONENT_MODE=--start' not in result.stdout and 'COMPONENT_MODE=--apply' not in result.stdout
