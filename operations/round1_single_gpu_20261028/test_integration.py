"""Real isolated supervisor and action subprocess, generated software fixtures.

No GPU, real EEG, real Slurm command, external credentials or network is used.
The scientific stand-in requires unprojected input and an actual first strict
read/validation before the unchanged supervisor can mark its action complete.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap


HERE = Path(__file__).resolve().parent
ORIGINAL = Path('/workspace/paper3-r1-full-repair/vendor')


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))


def test_actual_isolated_native_supervisor_science_child_return_and_no_projection_leak(tmp_path):
    root, home, native, bundle, direct = [tmp_path / name for name in
                                         ('project', 'home', 'native', 'bundle', 'direct')]
    for directory in (root, home, native, bundle, direct):
        directory.mkdir()
    runtime = root / 'runs/training_campaign_parallel_v1'
    runtime.mkdir(parents=True)
    old_cache = home / 'paper3-r1-full-repair/2bfd664251ee5d1bcaf5/verified_context_cache.py'
    write(old_cache, '''
        class ActionSnapshot:
            def __init__(self, root, action, emit=None): self.root=root
            def install(self, training, development): return self
            def checkpoint(self): pass
            def close(self, successful=True): pass
    ''')
    for name in ('worker_runner.py', 'single_gpu.py', 'receipt_projection.py', 'finalization_cache.py'):
        shutil.copyfile(HERE / name, bundle / name)
    # Execute actual deployed worker supervision and actual original scheduler
    # outcome writer; only the scientific fixture is replaced below.
    shutil.copyfile(ORIGINAL / 'worker.py', native / 'worker.py')
    shutil.copyfile(ORIGINAL / 'scheduler.py', native / 'scheduler.py')
    write(native / 'action_startup.py', '''
        from contextlib import contextmanager
        @contextmanager
        def active(root, action, **kwargs):
            yield
    ''')
    write(direct / 'direct_runner.py', '''
        import os
        def reuse_node_cache(root, prior_job): return []
        def child_environment(root, manifest, seconds, cache_paths=None):
            result=os.environ.copy()
            result['PYTHONPATH']=str(root)
            result['RETAINED_DRIVER_HOOK']='original-driver-overlay'
            return result
    ''')
    write(root / 'p3work/__init__.py', '')
    write(root / 'p3work/classical_training.py', '# No model loaded in this software fixture.\n')
    write(root / 'p3work/development_evaluation.py', '''
        import json,hashlib
        from pathlib import Path
        calls={'receipt':0,'configuration':0,'writer':0}
        def checked(root, descriptor):
            path=Path(root)/descriptor['path']
            data=path.read_bytes()
            if hashlib.sha256(data).hexdigest()!=descriptor['sha256']:
                raise ValueError('fixture strict artifact hash mismatch')
            value=json.loads(data)
            if value['scientific_fit_completed'] is not True:
                raise ValueError('fixture original scientific completion proof required')
            return value
        def verify_development_receipt(root, descriptor):
            calls['receipt']+=1
            value=checked(root,descriptor)
            assert value['science_only_payload']==['unprojected strict input']*2000
            return value
        def verify_configuration_receipt(root, descriptor):
            calls['configuration']+=1
            return verify_development_receipt(root,descriptor)
        def _put(path,value):
            calls['writer']+=1
            with Path(path).open('x') as out: json.dump(value,out)
    ''')
    write(native / 'kernel_bridge.py', '''
        import json,os,sys
        from pathlib import Path
        def _read(path,default=None):
            try:return json.loads(Path(path).read_text())
            except FileNotFoundError:return default
        def action_complete(root,action):
            return (_read(Path(root)/action['expected_receipt'],{}) or {}).get('status')=='STRONGEST_DEVELOPMENT_CONTROL_FROZEN'
        def perform_action(root,action,payload_manifest,*,allocation_seconds,direct_package,stop_requested,on_ready):
            assert 'receipt_projection' not in sys.modules, 'metadata projection leaked into scientific child'
            assert os.environ['RETAINED_DRIVER_HOOK']=='original-driver-overlay'
            data=_read(Path(root)/'scientific-input.json')
            assert 'science_only_payload' in data, 'scientific full reader was projected'
            import action_startup
            from p3work import development_evaluation as dev
            with action_startup.active(root,action):
                on_ready()
                ref=json.loads((Path(root)/'input-descriptor.json').read_text())
                first=dev.verify_configuration_receipt(root,ref)
                assert first['score']==0.625
                first['score']=999
                again=dev.verify_configuration_receipt(root,ref)
                third=dev.verify_development_receipt(root,ref)
                assert again['score']==third['score']==0.625
                assert dev.calls=={'receipt':1,'configuration':1,'writer':0}
                result={'status':'STRONGEST_DEVELOPMENT_CONTROL_FROZEN','fixture_only':True,'selected':'B4','score':again['score']}
                target=Path(root)/action['expected_receipt']; target.parent.mkdir(parents=True,exist_ok=True)
                dev._put(target,result)
                assert dev.calls['writer']==1
                (Path(root)/'runs/training_campaign_parallel_v1/queue.json').write_text(json.dumps({'status':'COMPLETE'}))
            assert dev.verify_development_receipt.__name__=='verify_development_receipt'
            return {'status':'COMPLETE','fixture_only':True}
    ''')
    science = dict(status='DEVELOPMENT_EVALUATION_COMPLETE', scientific_fit_completed=True,
                   science_only_payload=['unprojected strict input'] * 2000, score=.625)
    raw = json.dumps(science).encode()
    (root / 'scientific-input.json').write_bytes(raw)
    (root / 'input-descriptor.json').write_text(json.dumps(dict(path='scientific-input.json',
        sha256=hashlib.sha256(raw).hexdigest())))
    action = dict(kind='strongest_control', key='strongest-control', serial=True,
                  affinity_slot=0, expected_receipt='runs/training_campaign_v1/strongest_control.json')
    (runtime / 'queue.json').write_text(json.dumps(dict(status='RUNNING', ready_actions=[action], retired_slots=[])))
    (runtime / 'claims.json').write_text('{}')
    manifest = root / 'runtime-manifest.json'; manifest.write_text('{}')
    command = [sys.executable, '-I', '-S', '-B', '-u', str(bundle / 'worker_runner.py'),
               '--root', str(root), '--runtime-dir', str(runtime), '--slot', '0', '--package', str(native),
               '--payload-manifest', str(manifest), '--direct-package', str(direct),
               '--prior-job', 'old-fixture', '--allocation-seconds', '1000']
    env = dict(os.environ, HOME=str(home), SLURM_JOB_ID='991234', PAPER3_R1_MARKER='fixture-marker',
               PAPER3_R1_ATTEMPT_NAME='fixture-single-s0-a1')
    process = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
    assert process.returncode == 0, process.stdout + '\n' + process.stderr
    result = json.loads((root / action['expected_receipt']).read_text())
    assert result == dict(status='STRONGEST_DEVELOPMENT_CONTROL_FROZEN', fixture_only=True,
                          selected='B4', score=.625)
    assert json.loads((runtime / 'claims.json').read_text()) == {}
    outcome = json.loads((runtime / 'action-outcomes' /
        (hashlib.sha256(action['key'].encode()).hexdigest() + '-991234.json')).read_text())
    assert outcome['receipt_complete'] is True and outcome['returncode'] == 0
    assert outcome['returned']['status'] == 'COMPLETE'
    worker_outcome = json.loads((runtime / 'scheduler-outcomes/991234.json').read_text())
    assert worker_outcome['exit_code'] == 0 and worker_outcome['marker'] == 'fixture-marker'
    assert 'R1_FINALIZATION_VERIFICATION_REUSE_ACTIVE' in process.stdout
    assert 'R1_FINALIZATION_DEPENDENCIES_CHECKED_BEFORE_PUBLICATION' in process.stdout
    assert 'R1_FINALIZATION_NATIVE_VERIFICATION_FINISHED' in process.stdout
    assert 'R1_ACTION_FINISHED' in process.stdout
    assert raw == (root / 'scientific-input.json').read_bytes()
