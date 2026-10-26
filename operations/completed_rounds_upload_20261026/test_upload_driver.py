"""Driver guarantees: detached lifetime, deduplication and partial upload semantics."""
from argparse import Namespace
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

HERE=Path(__file__).resolve().parent


def load():
    spec=importlib.util.spec_from_file_location('completed_upload_driver_test',HERE/'upload_driver.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_detached_workers_survive_parent_and_duplicate_start_reuses_locks(tmp_path):
    package=tmp_path/'package';package.mkdir();stub=package/'upload_driver.py'
    stub.write_text('''import argparse,fcntl,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--channel');p.add_argument('--root');p.add_argument('--work-dir');p.add_argument('--rounds',nargs='+');a=p.parse_args()
w=Path(a.work_dir)/a.channel
with (w/'driver.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (w/'worker.json').write_text(json.dumps({'pid':os.getpid(),'channel':a.channel}))
 while True:
  (w/'heartbeat').write_text(str(time.time()));time.sleep(.05)
''')
    code="""import importlib.util,pathlib,sys
from argparse import Namespace
s=importlib.util.spec_from_file_location('real_driver',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.HERE=pathlib.Path(sys.argv[2]);m.start(Namespace(work_dir=sys.argv[3],rounds=[4,5,6,8],root=sys.argv[4],credentials=None))
"""
    command=[sys.executable,'-I','-S','-B','-c',code,str(HERE/'upload_driver.py'),str(package),str(tmp_path/'work'),str(tmp_path/'unused-source')]
    children=[]
    try:
        first=subprocess.run(command,capture_output=True,text=True,timeout=12)
        assert first.returncode==0,first.stdout+first.stderr
        started=json.loads((tmp_path/'work/STARTED.json').read_text())
        assert started['terminal_can_close'] is True
        children=[row['pid'] for row in started['processes']]
        assert len(children)==2 and all(os.getsid(pid)==pid for pid in children)
        time.sleep(.12)
        assert all((Path('/proc')/str(pid)).exists() for pid in children)
        second=subprocess.run(command,capture_output=True,text=True,timeout=12)
        assert second.returncode==0,second.stdout+second.stderr
        duplicate=json.loads((tmp_path/'work/STARTED.json').read_text())
        assert all(row['reused'] for row in duplicate['processes'])
        assert [row['worker']['pid'] for row in duplicate['processes']]==children
        assert not (tmp_path/'unused-source').exists()
    finally:
        for pid in children:
            try:
                argv=(Path('/proc')/str(pid)/'cmdline').read_bytes()
                if str(stub).encode() in argv:
                    os.kill(pid,signal.SIGTERM)
            except (OSError,ProcessLookupError):
                pass


def test_partial_pointer_never_marks_round_done(tmp_path,monkeypatch):
    driver=load()
    sys.path.insert(0,str(HERE))
    import collector
    import paper_credentials
    import transport.lossless_publish as transfer
    root=tmp_path/'root';output=root/'runs/followup_20261024_r4';output.mkdir(parents=True)
    raw=b'{"recipe_sha256":"fixture"}\n';(output/'plan.json').write_bytes(raw)
    stage=tmp_path/'stage';stage.mkdir();(stage/'plan.json').write_bytes(raw)
    bundle={'source_base':stage,'paths':['plan.json'],'metadata':{'collection_status':'COMPLETE',
        'plan_sha256':hashlib.sha256(raw).hexdigest(),'report_inventory':[], 'reports':[],
        'all_complete':True,'required_missing':[]}}
    collected=[];published=[]
    def collect(*args,**kwargs):collected.append(True);return bundle
    class Git:
        def __init__(self,*a,**kw):pass
        branch='paper3-results'
        prefix='repos/test/repo'
        def api(self,*a,**kw):
            return {'content':base64.b64encode(json.dumps({'snapshot_id':'old',
                'status':'PAPER_DATA_FILES_VERIFIED'}).encode()).decode()}
    def publish(*a,**kw):
        published.append(True)
        receipt={'collection_status':'COMPLETE','snapshot_id':'new',
            'status':'PAPER_DATA_PARTIAL_FILES_VERIFIED' if len(published)==1 else 'PAPER_DATA_FILES_VERIFIED'}
        driver.atomic(Path(a[3])/'round-04-PAPER-UPLOADED.json',receipt)
        return receipt
    monkeypatch.setattr(collector,'collect_round',collect)
    monkeypatch.setattr(paper_credentials,'upload_discover_token',lambda *a:'test-token')
    monkeypatch.setattr(transfer,'VerifiedGitHub',Git)
    monkeypatch.setattr(transfer,'stage_file',lambda *a,**kw:{'source_path':'plan.json'})
    monkeypatch.setattr(transfer,'publish_snapshot',publish)
    monkeypatch.setattr(driver.time,'sleep',lambda n:None)
    monkeypatch.setattr(driver.os,'nice',lambda n:None)
    driver.atomic(tmp_path/'work/round-04/round-04-PAPER-UPLOADED.json',
        {'snapshot_id':'old','status':'PAPER_DATA_FILES_VERIFIED','collection_status':'COMPLETE'})
    driver.paper_worker(Namespace(root=str(root),rounds=[4],credentials=None),tmp_path/'work')
    assert len(published)==2 and len(collected)==1
    status=json.loads((tmp_path/'work/status.json').read_text())
    assert status['completed_rounds']==[4] and status['status']=='COMPLETE'


def test_replaced_or_new_reports_invalidate_cached_bundle(tmp_path):
    driver=load();root=tmp_path/'root';output=root/'runs/followup_20261024_r4';reports=output/'reports';reports.mkdir(parents=True)
    raw=b'{}';(output/'plan.json').write_bytes(raw);report=reports/'report.json';report.write_bytes(b'old')
    def sig():
        s=report.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
    bundle={'metadata':{'plan_sha256':hashlib.sha256(raw).hexdigest(),
        'reports':[{'source_path':'runs/followup_20261024_r4/reports/report.json','source_signature':sig()}],
        'report_inventory':['runs/followup_20261024_r4/reports/report.json']}}
    assert driver.source_plan_unchanged(bundle,root,4)
    (reports/'later.csv').write_text('a,b\n1,2\n')
    assert not driver.source_plan_unchanged(bundle,root,4)
    (reports/'later.csv').unlink();report.write_bytes(b'new')
    assert not driver.source_plan_unchanged(bundle,root,4)
