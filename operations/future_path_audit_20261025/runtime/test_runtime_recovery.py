import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('resume_existing',HERE/'resume_existing.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def project(tmp_path,number):
    root=tmp_path/'project';out=root/'runs'/R.OUTPUTS[number]
    op=tmp_path/'native-operation';op.mkdir(exist_ok=True)
    (op/'coordinator.py').write_text('''import json
def prepare_context(*a,**k):raise AssertionError("Never rebuild launch")
def controller_loop(c,**kwargs):
 (c["runtime"]/"native-controller-called.json").write_text(json.dumps({"launch":c["launch"],"plan":c["plan"],"operation":"existing_controller_loop"}))
 return 0
''')
    (op/'monitor.py').write_text('raise AssertionError("not started in reused test")\n')
    (op/'result_channels.py').write_text('raise AssertionError("not started in reused test")\n')
    plan={'root':str(root),'output':str(out),'recipe_sha256':'original-recipe','tasks':[{'key':'a'}]}
    save(out/'plan.json',plan)
    launch={'round':number,'root':str(root),'output':str(out),'plan':str(out/'plan.json'),
            'worker_command':[sys.executable,str(op/'three_jobs.py'),'worker','--slot','{slot}'],
            'partition':'hpc_gpu1' if number in (5,6) else 'hpc_gpu',
            'cpus':12 if number in (5,6) else 16,'memories':['64G']*3,'gres':'gpu:1',
            'python':sys.executable,'package':str(tmp_path/'native-source'),
            'original_plan_sha256':hashlib.sha256((out/'plan.json').read_bytes()).hexdigest(),
            'old_job':'772012' if number==5 else '772013','campaign_identity':'original-recipe'}
    save(out/'runtime/three_jobs/launch.json',launch)
    save(out/'a/result.json',{'status':'COMPLETE','task':{'key':'a'}})
    (out/'a/checkpoint.pt').write_bytes(b'existing-checkpoint\x00')
    return root,out,launch


@pytest.mark.parametrize('number',[5,6,8,9])
def test_real_isolated_resume_calls_installed_loop_preserving_saved_launch(tmp_path,number):
    root,out,launch=project(tmp_path,number)
    before={str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}
    result=subprocess.run([sys.executable,'-I','-S','-B','-u',str(HERE/'resume_existing.py'),
        '--round',str(number),'--root',str(root),'--run-controller'],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
    value=json.loads((out/'runtime/three_jobs/native-controller-called.json').read_text())
    assert value['launch']==launch
    assert value['plan']['recipe_sha256']=='original-recipe'
    assert all(Path(path).read_bytes()==data for path,data in before.items())


@pytest.mark.parametrize('number',[5,6,8,9])
def test_live_controller_lock_is_reused_without_any_process_or_job_action(tmp_path,number,monkeypatch):
    import fcntl
    root,out,launch=project(tmp_path,number);c=R.context(root,number)
    def forbidden(*a,**k):raise AssertionError('Must immediately reuse live controller')
    with (c['runtime']/'controller.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert R.start(c,'controller',popen=forbidden)=={'kind':'controller','reused':True}
    assert json.loads((c['runtime']/'launch.json').read_text())==launch


def test_existing_plan_launch_mismatch_never_rebuilds_or_overwrites(tmp_path):
    root,out,_=project(tmp_path,9)
    plan=out/'plan.json';original=plan.read_bytes();plan.write_bytes(original+b' ')
    before=plan.read_bytes()
    with pytest.raises(ValueError,match='Saved launch'):
        R.context(root,9)
    assert plan.read_bytes()==before


def test_no_installer_hash_bypass_or_compute_import_in_preparation(tmp_path):
    root,out,launch=project(tmp_path,9)
    result=subprocess.run([sys.executable,'-I','-S','-u','-c',
        'import runpy,sys;sys.argv='+repr([str(HERE/'resume_existing.py'),'--root',str(root),'--round','9'])+
        ';runpy.run_path('+repr(str(HERE/'resume_existing.py'))+',run_name="__main__")'],
        capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'EXISTING_PLAN_RESUME_READY' in result.stdout
    assert 'installer_reentered": false' in result.stdout


@pytest.mark.parametrize('number',[5,6,8,9])
def test_actual_native_controller_accepts_saved_context_without_queue_actions(tmp_path,number):
    root,out,launch=project(tmp_path,number)
    op=(Path('/tmp/paper3-r56-threejobs-build') if number in (5,6) else
        Path('/workspace/paper3-round%d/runtime'%number))
    launch['worker_command']=[sys.executable,str(op/'three_jobs.py'),'worker','--slot','{slot}']
    launch['account']='fixture';launch['old_partition']='hpc_gpu'
    save(out/'runtime/three_jobs/launch.json',launch)
    before_plan=(out/'plan.json').read_bytes()
    script='''import importlib.util,json,pathlib,sys,threading
helper_path,project,number=sys.argv[1:]
spec=importlib.util.spec_from_file_location("existing_resume",helper_path)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
c=helper.context(project,int(number));sys.path.insert(0,str(c["operation"]))
spec=importlib.util.spec_from_file_location("native_controller",c["operation"]/"coordinator.py")
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
def forbidden(*args,**kwargs):raise AssertionError("No Slurm command from stopped native controller")
stop=threading.Event();stop.set()
answer=native.controller_loop({k:v for k,v in c.items() if k not in ("operation","plan_bytes","launch_bytes")},stop=stop,runner=forbidden)
assert answer==0
assert "torch" not in sys.modules
print("ACTUAL_NATIVE_CONTEXT_ACCEPTED")
'''
    result=subprocess.run([sys.executable,'-I','-S','-u','-c',script,
        str(HERE/'resume_existing.py'),str(root),str(number)],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'ACTUAL_NATIVE_CONTEXT_ACCEPTED' in result.stdout
    assert 'RETRY' not in result.stdout,result.stdout
    assert json.loads((out/'runtime/three_jobs/launch.json').read_text())==launch
    assert (out/'plan.json').read_bytes()==before_plan


@pytest.mark.parametrize('number',[5,6,8,9])
def test_execute_reuses_every_running_component_without_spawn(tmp_path,number,monkeypatch):
    import fcntl
    from contextlib import ExitStack
    root,out,launch=project(tmp_path,number)
    runtime=out/'runtime/three_jobs'
    components=('controller','monitor','paper','archive') if number==9 else ('controller','monitor')
    def forbidden(*args,**kwargs):raise AssertionError('All components already hold native locks')
    monkeypatch.setattr(R.subprocess,'Popen',forbidden)
    before={str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}
    with ExitStack() as stack:
        for kind in components:
            handle=stack.enter_context((runtime/(kind+'.lock')).open('a'))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert R.main(['--round',str(number),'--root',str(root),'--execute'])==0
    assert all(Path(path).read_bytes()==data for path,data in before.items())


def test_post_repair_original_pinned_installer_rejects_its_changed_operational_backend(tmp_path):
    path=Path('/workspace/paper3-r9-thin-delivery/START-PAPER3-R9-SMALL.py')
    spec=importlib.util.spec_from_file_location('frozen_r9_thin',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    installed=module.install(tmp_path/'installed')
    assert module.verify_install(installed)==installed
    backend=installed/'runtime/backend_entry.py'
    old=backend.read_bytes();backend.write_bytes(old+b'\n# installed runtime repair\n')
    with pytest.raises(RuntimeError,match='Installed file hash mismatch.*runtime/backend_entry.py'):
        module.install(tmp_path/'installed')
    # The installed entry remains available for saved-plan resume; immutable
    # science source was not overwritten by the old installer.
    assert backend.read_bytes()==old+b'\n# installed runtime repair\n'
    assert (installed/'START.py').is_file()
