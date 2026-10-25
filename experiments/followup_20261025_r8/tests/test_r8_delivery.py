import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import socket
import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('r8_make_launch',ROOT/'make_launch.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_resume_failover_hash_check_and_cached_restart(tmp_path):
    data=(b'round8-cached-data-'*5000)
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
        assert ('/good','bytes=10000-') in calls
        prior=len(calls)
        subprocess.run(['bash','-c',command],check=True,capture_output=True,timeout=20)
        assert len(calls)==prior
        final.unlink()
        (tmp_path/'download.py.part').write_bytes(b'bad-existing-partial')
        command=module.download_command(final,digest,[base+'/missing',base+'/tampered',base+'/good'],retry_seconds=0)
        result=subprocess.run(['bash','-c',command],capture_output=True,text=True,timeout=20)
        assert result.returncode==0,result.stderr
        assert final.read_bytes()==data
        assert not (tmp_path/'download.py.part').exists()
    finally:
        server.shutdown();server.server_close();thread.join(2)


def test_school_command_single_line_and_fixed_resources():
    command=module.school_command('a'*40,{'wrapper_sha256':'b'*64})
    assert '\n' not in command
    assert '--submit' in command
    assert 'raw.githubusercontent.com' in command and 'api.github.com' in command
    assert ' -k ' not in command
    assert ' -C - ' in command
    assert '/botocore/cacert.pem' in command
    subprocess.run(['bash','-n','-c',command],check=True)
