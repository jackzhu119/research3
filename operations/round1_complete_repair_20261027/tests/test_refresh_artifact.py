"""Exercise the exact generated standalone entry under isolated Python."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_refresh_hotfix


def test_generated_entry_default_is_readonly_with_real_original_package(tmp_path):
    source = tmp_path/'build'; source.mkdir()
    for name in ('supervisor.py', 'refresh_hotfix_template.py'):
        (source/name).write_bytes((ROOT/name).read_bytes())
    result = build_refresh_hotfix.build(source)
    entry = source/'delivery'/result['entry']
    assert hashlib.sha256(entry.read_bytes()).hexdigest() == result['entry_sha256']
    spec = importlib.util.spec_from_file_location('r1_generated_refresh_entry', entry)
    generated = importlib.util.module_from_spec(spec); spec.loader.exec_module(generated)
    home = tmp_path/'home'; root = tmp_path/'project'; root.mkdir()
    old = home/'paper3-r1-full-repair'/generated.OLD_PACKAGE
    work = old.parent/'work'; old.mkdir(parents=True); work.mkdir()
    with zipfile.ZipFile(ROOT/'delivery/paper3-r1-complete-repair.zip') as archive:
        for name in ('MANIFEST.json', 'supervisor.py', *generated.DEPENDENCIES):
            (old/name).write_bytes(archive.read(name))
    token = 'artifact-original-manager-nonce'
    argv = [sys.executable, '-I', '-S', '-B', '-u', str(old/'supervisor.py'), 'manage',
        '--root', str(root), '--home', str(home), '--package', str(old), '--work', str(work),
        '--submit', '--launch-token', token, '--python', sys.executable]
    (work/'controller-process.json').write_text(json.dumps(dict(pid=74123, root=str(root), launch_token=token)))
    before = {str(p): p.read_bytes() for p in home.rglob('*') if p.is_file()}
    record = generated.apply(home, root, process=lambda pid: dict(pid=pid, uid=os.getuid(),
        start_ticks='artifact-fixture-start', argv=argv),
        send=lambda *a: (_ for _ in ()).throw(AssertionError('inspection sent a signal')),
        popen=lambda *a, **k: (_ for _ in ()).throw(AssertionError('inspection spawned a process')))
    assert record['status'] == 'READ_ONLY_HOTFIX_READY'
    assert record['automatic_child_refresh'] is False
    assert before == {str(p): p.read_bytes() for p in home.rglob('*') if p.is_file()}
    harness = r'''import runpy,sys
namespace=runpy.run_path(sys.argv[1],run_name='standalone_artifact_audit')
assert not any(name.split('.')[0] in {'torch','numpy','scipy','sklearn'} for name in sys.modules)
assert namespace['SUPERVISOR_SHA256']==sys.argv[2]
print('ISOLATED_STANDALONE_ENTRY_IMPORT_OK')
'''
    checked = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', harness, str(entry),
        result['supervisor_sha256']], capture_output=True, text=True, timeout=10)
    assert checked.returncode == 0, checked.stderr
    assert checked.stdout.strip() == 'ISOLATED_STANDALONE_ENTRY_IMPORT_OK'
