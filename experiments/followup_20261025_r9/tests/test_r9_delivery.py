import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import socket
import subprocess
import threading
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('r9_make_launch',ROOT/'make_launch.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_resume_failover_hash_check_and_cached_restart(tmp_path):
    data=(b'round9-cached-data-'*5000)
    calls=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_): pass
        def do_GET(self):
            calls.append((self.path,self.headers.get('Range')))
            if self.path=='/missing':
                self.send_error(404);return
            if self.path=='/tampered':
                bad=b'not-the-package'
                self.send_response(200);self.send_header('Content-Length',str(len(bad)))
                self.end_headers();self.wfile.write(bad);return
            start=int(self.headers.get('Range','bytes=0-').split('=')[1].split('-')[0])
            if start>=len(data):
                self.send_error(416);return
            self.send_response(206 if start else 200)
            if start:self.send_header('Content-Range',f'bytes {start}-{len(data)-1}/{len(data)}')
            self.send_header('Content-Length',str(len(data)-start));self.end_headers()
            if self.path=='/interrupted' and start==0:
                self.wfile.write(data[:10000]);self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR);self.connection.close();return
            self.wfile.write(data[start:])
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        base=f'http://127.0.0.1:{server.server_port}'
        final=tmp_path/'download.py'
        digest=hashlib.sha256(data).hexdigest()
        command=module.download_command(final,digest,[base+'/interrupted',base+'/good'],retry_seconds=0)
        subprocess.run(['bash','-n','-c',command],check=True)
        result=subprocess.run(['bash','-c',command],capture_output=True,text=True,timeout=20)
        assert result.returncode==0,result.stderr
        assert final.read_bytes()==data
        assert ('/good',None) in calls
        prior=len(calls)
        subprocess.run(['bash','-c',command],check=True,capture_output=True,timeout=20)
        assert len(calls)==prior
        final.unlink()
        (tmp_path/'download.py.part.2').write_bytes(b'bad-existing-partial')
        command=module.download_command(final,digest,[base+'/missing',base+'/tampered',base+'/good'],retry_seconds=0)
        result=subprocess.run(['bash','-c',command],capture_output=True,text=True,timeout=20)
        assert result.returncode==0,result.stderr
        assert final.read_bytes()==data
        assert not (tmp_path/'download.py.part.2').exists()
    finally:
        server.shutdown();server.server_close();thread.join(2)


def test_school_command_single_line_and_fixed_resources():
    command=module.school_command('a'*40,{'wrapper_sha256':'b'*64})
    assert '\n' not in command
    assert '--prepare-only' in command and '--submit' not in command
    assert '--submit' in module.school_command('a'*40,{'wrapper_sha256':'b'*64},submit=True)
    assert 'raw.githubusercontent.com' in command and 'api.github.com' in command
    assert ' -k ' not in command
    assert ' -C - ' in command
    assert '/botocore/cacert.pem' in command
    assert 'T="$F.part.$I"' in command
    subprocess.run(['bash','-n','-c',command],check=True)


def test_backup_range_rejection_preserves_slow_raw_resume(tmp_path):
    data=b'resumable-raw-download-'*7000
    calls=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def do_GET(self):
            range_header=self.headers.get('Range')
            calls.append((self.path,range_header))
            start=int((range_header or 'bytes=0-').split('=')[1].split('-')[0])
            if self.path=='/api-no-range':
                self.send_response(200)
                self.send_header('Content-Length',str(len(data)))
                self.end_headers()
                try:self.wfile.write(data)
                except (BrokenPipeError,ConnectionResetError):pass
                return
            self.send_response(206 if start else 200)
            if start:self.send_header('Content-Range',f'bytes {start}-{len(data)-1}/{len(data)}')
            self.send_header('Content-Length',str(len(data)-start));self.end_headers()
            if not start:
                self.wfile.write(data[:12000]);self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR);self.connection.close()
                return
            self.wfile.write(data[start:])
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    final=tmp_path/'large.py'
    # Exercise curl code33 on the backup while the raw endpoint has useful data.
    (tmp_path/'large.py.part.2').write_bytes(data[:4000])
    try:
        base=f'http://127.0.0.1:{server.server_port}'
        command=module.download_command(final,hashlib.sha256(data).hexdigest(),
            [base+'/slow-raw',base+'/api-no-range'],retry_seconds=0)
        result=subprocess.run(['bash','-c',command],capture_output=True,text=True,timeout=20)
        assert result.returncode==0,result.stderr
        assert final.read_bytes()==data
        assert ('/api-no-range','bytes=4000-') in calls
        assert ('/slow-raw','bytes=12000-') in calls
        assert sum(p=='/slow-raw' and h is None for p,h in calls)==1
    finally:
        server.shutdown();server.server_close();thread.join(2)


def test_builder_excludes_downloaded_evidence_and_retains_credential_code():
    selected_spec=importlib.util.spec_from_file_location('r9_builder',ROOT/'build_online.py')
    builder=importlib.util.module_from_spec(selected_spec);selected_spec.loader.exec_module(builder)
    assert builder._selected(Path('runtime/paper_credentials.py'))
    assert not builder._selected(Path('private/github.json'))
    assert not builder._selected(Path('evidence/round-07/download/report.json'))
    assert not builder._selected(Path('evidence/round-03/plan.json'))
    assert builder._selected(Path('evidence/RESULTS_FOR_R9_ZH.md'))
    assert builder._selected(Path('evidence/FINDINGS.json'))
    assert builder._selected(Path('data/pan2023/some/X.npy'))
    assert builder._selected(Path('data/weibo/some/events.json'))


def test_actual_payload_isolated_install_cached_restart_and_no_submit(tmp_path):
    selected_spec=importlib.util.spec_from_file_location('r9_builder_full',ROOT/'build_online.py')
    builder=importlib.util.module_from_spec(selected_spec);selected_spec.loader.exec_module(builder)
    wrapper=tmp_path/'START-PAPER3-R9-ONLINE.py'
    manifest=builder.build(ROOT,wrapper,tmp_path/'PACKAGE.json')
    assert manifest['round']==9 and manifest['default_submission'] is False
    assert manifest['uncompressed_payload_bytes'] <= 64*1024**2
    assert manifest['compute_stage_downloads'] is False
    assert 'runtime/result_channels.py' in manifest['files']
    assert 'runtime/paper_credentials.py' in manifest['files']
    assert any(p.startswith('data/pan2023/') and p.endswith('.npy') for p in manifest['files'])
    assert any(p.startswith('data/weibo/') and p.endswith('.npy') for p in manifest['files'])
    assert not any(p.startswith('evidence/round-') for p in manifest['files'])
    install=tmp_path/'installed'
    command=[sys.executable,'-I','-S','-B','-u',str(wrapper),'--install-base',str(install),'--install-only']
    first=subprocess.run(command,capture_output=True,text=True,timeout=60)
    assert first.returncode==0, first.stderr
    receipt=json.loads(first.stdout.splitlines()[0])
    package=Path(receipt['package'])
    before=(package/'INSTALL_MANIFEST.json').stat().st_mtime_ns
    again=subprocess.run(command,capture_output=True,text=True,timeout=60)
    assert again.returncode==0,again.stderr
    assert (package/'INSTALL_MANIFEST.json').stat().st_mtime_ns==before
    # Even if sbatch exists, default/prepare mode cannot reach it.
    fakebin=tmp_path/'bin';fakebin.mkdir()
    called=tmp_path/'sbatch-called'
    fake=fakebin/'sbatch'
    fake.write_text('#!/bin/sh\nprintf called > '+str(called)+'\nexit 88\n');fake.chmod(0o700)
    root=tmp_path/'project';root.mkdir()
    env=dict(os.environ,PATH=str(fakebin)+os.pathsep+os.environ.get('PATH',''))
    prepare=[sys.executable,'-I','-S','-B','-u',str(wrapper),'--install-base',str(install),
             '--prepare-only','--root',str(root),'--python',sys.executable]
    result=subprocess.run(prepare,capture_output=True,text=True,env=env,timeout=60)
    assert result.returncode==0,result.stderr
    assert 'ROUND_9_THREE_JOBS_PREPARED' in result.stdout
    assert not called.exists()
    plan_path=root/'runs/followup_20261025_r9/plan.json'
    plan=json.loads(plan_path.read_text())
    assert plan['resources']['jobs']==3
    assert plan['resources']['gpus_per_job']==1
    assert plan['resources']['cpus_per_job']==16
    assert plan['resources']['memory_per_job']=='64G'
    assert plan['resources']['partition']=='hpc_gpu'
    assert len(plan['tasks']) > 0
    # Default forwarding has the same no-submit behavior as explicit prepare.
    default=[x for x in prepare if x!='--prepare-only']
    result=subprocess.run(default,capture_output=True,text=True,env=env,timeout=60)
    assert result.returncode==0,result.stderr
    assert not called.exists()
