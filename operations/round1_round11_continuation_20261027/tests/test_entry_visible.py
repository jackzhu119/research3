import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('visible_diagnostic_test_entry',HERE/'entry.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

def test_missing_task_observation_is_unknown():
    assert entry.summarize_progress({'value':{}})['task_observation_valid'] is False
    assert entry.summarize_progress({'value':{'tasks':None}})['counts'] is None

def test_completed_and_retained_error_are_separate():
    row={'value':{'tasks':[{'key':'a','status':'COMPLETE'},
        {'key':'profile-B5','status':'RUNNING','error_type':'ExitCode75'}]}}
    s=entry.summarize_progress(row)
    assert s['counts']=={'COMPLETE':1,'RUNNING':1}
    assert s['active_or_retained_error_tasks'][0]['error_type']=='ExitCode75'

def test_owned_reader_timeout_is_bounded():
    started=time.monotonic()
    out=entry.bounded_command([sys.executable,'-I','-S','-B','-c','import time;time.sleep(10)'],seconds=.1)
    assert out['code']==124 and time.monotonic()-started<2

def test_reader_limits_and_symlink_guard(tmp_path):
    root=tmp_path/'root';root.mkdir();outside=tmp_path/'private';outside.write_text('DO_NOT_READ')
    (root/'escape').symlink_to(outside)
    assert entry.read_file(root/'escape',(root,))['status']=='UNAVAILABLE'
    (root/'big').write_bytes(b'x'*100)
    assert entry.read_file(root/'big',(root,),cap=10)['oversized']

def test_collect_is_readonly_and_has_no_wait(tmp_path,monkeypatch):
    root=tmp_path/'root';home=tmp_path/'home';root.mkdir();home.mkdir()
    runtime=root/'runs/training_campaign_parallel_v1'
    write(runtime/'progress.json',{'tasks':[{'key':'a','status':'COMPLETE'}]})
    for i in range(3):write(runtime/'worker-status'/('slot-%d.json'%i),{'slot':i})
    before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    started=time.monotonic()
    result=entry.collect(root,home)
    after={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert before==after and result['scientific_jobs_signalled']==0 and result['jobs_submitted']==0
    assert time.monotonic()-started<15

def test_start_message_precedes_collection_failure(tmp_path,monkeypatch):
    def fail(*_):raise RuntimeError('fixture')
    monkeypatch.setattr(entry,'collect',fail)
    output=io.StringIO()
    try:
        with contextlib.redirect_stdout(output):entry.main(['--no-upload','--root',str(tmp_path),'--home',str(tmp_path)])
    except RuntimeError:pass
    assert '"stage": "START"' in output.getvalue()
