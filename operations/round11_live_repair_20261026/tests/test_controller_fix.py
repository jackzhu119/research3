import hashlib,importlib,json,os
from pathlib import Path
import subprocess,sys
from types import SimpleNamespace
import pytest
from controller_fix import bounded_read,install_read_patch,install_plan_cache,load_original,validate_operational_history,controller_loop,scheduler_identity

ORIGINAL=Path(os.environ.get('PAPER3_R11_TEST_ORIGINAL_PACKAGE','/workspace/paper3-round11'))


def test_tiny_json_old64m_preallocation_fails_under_real_address_limit_new_reader_passes(tmp_path):
    p=tmp_path/'tiny.json';p.write_text('{"saved":true}')
    script='''import json,resource,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]);from controller_fix import bounded_read
resource.setrlimit(resource.RLIMIT_AS,(64*1024*1024,64*1024*1024))
try:
 with Path(sys.argv[2]).open('rb') as h:h.read(64*1024*1024+1)
except MemoryError:old='MemoryError'
else:raise AssertionError('Expected old implementation allocation failure')
assert bounded_read(sys.argv[2],None,max_bytes=64*1024*1024)=={'saved':True}
print(json.dumps({'old':old,'new':'OK'}))
'''
    done=subprocess.run([sys.executable,'-I','-S','-B','-c',script,str(Path(__file__).resolve().parents[1]),str(p)],capture_output=True,text=True)
    assert done.returncode==0,done.stderr
    assert json.loads(done.stdout)=={'old':'MemoryError','new':'OK'}


def test_safe_read_preserves_limit_corrupt_and_missing_defaults(tmp_path):
    p=tmp_path/'value';p.write_bytes(b'{"n":2}')
    assert bounded_read(p,max_bytes=8)=={'n':2}
    assert bounded_read(p,'OVER',max_bytes=6)=='OVER'
    p.write_bytes(b'not-json');assert bounded_read(p,'BAD')=='BAD'
    assert bounded_read(tmp_path/'missing','MISSING')=='MISSING'


def fixture(tmp_path):
    if str(ORIGINAL) not in sys.path:sys.path.insert(0,str(ORIGINAL))
    c=importlib.import_module('paper3r11.platform.common')
    root=tmp_path/'project';runtime=root/'runs/r11-repair-fixture/runtime';runtime.mkdir(parents=True)
    plan=dict(schema_version='paper3-r11-plan-v1',round=11,stage='campaign',root=str(root),package=str(ORIGINAL),
      output=str(runtime.parent),python=sys.executable,synthetic_fixture=True,
      code_manifest={'paper3r11/platform/common.py':c.sha(ORIGINAL/'paper3r11/platform/common.py')},
      resources=dict(jobs=4,gpus_per_job=1,cpus_per_job=16,memory_per_job='64G',partition='hpc_gpu',poll_seconds=.01),
      tasks=[dict(key='software-fixture-only',kind='fixture',role='meta_validation',requires=[],config={})])
    plan=c.sign_plan(plan);path=root/'plan.json';c.atomic(path,plan)
    env=load_original(ORIGINAL,path,expected_recipe=None);rt=env['runtime_module'];ctx=rt.context(path)
    c.atomic(runtime/'launch.json',ctx['launch']);c.atomic(runtime/'tasks.json',dict(recipe_sha256=plan['recipe_sha256'],tasks={}))
    c.atomic(runtime/'SUBMIT_AUTHORIZATION.json',dict(explicit_submit=True,plan_sha256=c.sha(path),recipe_sha256=plan['recipe_sha256']))
    marker=scheduler_identity(runtime,plan['recipe_sha256']);slots={}
    for i in range(4):
      a=dict(slot=i,sequence=1,marker=marker,name=marker+f'-s{i}-a1',job_id=str(775286+i),status='ACKNOWLEDGED',intent_at=1)
      slots[str(i)]=dict(attempts=[a],next_submit_at=0,rejections=0,worker_failures=0)
    c.atomic(runtime/'scheduler-state.json',dict(version=1,marker=marker,created_at=1,slots=slots))
    return c,env,path,plan


def test_cache_is_one_exact_object_and_changed_content_rejected(tmp_path):
    c,env,path,plan=fixture(tmp_path)
    first=c.load_plan(path)[1];second=env['runtime_module'].load_plan(path,validate_science_stage=False)[1]
    assert first is second
    data=path.read_bytes();os.utime(path,None);assert c.load_plan(path)[1] is first
    path.write_bytes(data+b' ')
    with pytest.raises(ValueError,match='AUTHORIZED_PLAN_CHANGED'):c.load_plan(path)


def test_existing_four_jobs_first_observed_never_resubmitted_and_science_untouched(tmp_path):
    c,env,path,plan=fixture(tmp_path);runtime=env['runtime'];marker=scheduler_identity(runtime,plan['recipe_sha256']);commands=[]
    code_before={p:c.sha(p) for p in ORIGINAL.rglob('*.py') if '__pycache__' not in p.parts}
    task_before=(runtime/'tasks.json').read_bytes();auth_before=(runtime/'SUBMIT_AUTHORIZATION.json').read_bytes();plan_before=path.read_bytes()
    def runner(command,**kwargs):
      commands.append(command)
      assert command[0] in ('squeue','sacct')
      lines='\n'.join(f'{775286+i}|{marker}|RUNNING|{marker}-s{i}-a1|gpu-node|gpu-node' for i in range(4)) if command[0]=='squeue' else ''
      return SimpleNamespace(returncode=0,stdout=lines,stderr='')
    result=controller_loop(path,tmp_path/'repair-work',package=ORIGINAL,submit=True,runner=runner,once=True,expected_recipe=None)
    assert result['inflight_jobs']==4 and {x[0] for x in commands}=={'squeue','sacct'}
    assert (runtime/'tasks.json').read_bytes()==task_before and (runtime/'SUBMIT_AUTHORIZATION.json').read_bytes()==auth_before and path.read_bytes()==plan_before
    assert all(c.sha(p)==h for p,h in code_before.items())
    proof=bounded_read(tmp_path/'repair-work/first-observation.json');assert proof['first_cycle_submit_allowed'] is False
    ready=bounded_read(tmp_path/'repair-work/controller-ready.json');assert ready['first_slurm_observation_completed'] and ready['memory_patch_loaded']


def test_missing_or_corrupt_scheduler_history_does_not_reset_or_submit(tmp_path):
    c,env,path,plan=fixture(tmp_path);file=env['runtime']/'scheduler-state.json';file.write_text('bad-state')
    before=file.read_bytes()
    with pytest.raises(ValueError,match='HISTORY_REQUIRED'):validate_operational_history(env)
    assert file.read_bytes()==before
