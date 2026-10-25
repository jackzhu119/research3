"""Run the exact native prepare_f0 dispatch branch with two real GPU profiles."""
import ast
from contextlib import nullcontext
import hashlib
import json
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import r1_spawn_bootstrap as boot


SOURCE=Path('/workspace/paper3_hpc/p3work/source_extension_downstream.py')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def profile(uuid):
    return {'python':'3.12.3','torch':'2.5.1+cu121','cuda':'12.1','device_name':'NVIDIA A100-SXM4-40GB',
        'device_uuid':uuid,'device_total_memory_bytes':42298834944,'device_compute_capability':[8,0],
        'driver_versions':['535.104.12']}


def fixture(tmp_path,monkeypatch,corrected=False):
    old=profile('b4f600a0-a5a4-b950-504b-b8c069e3e812')
    actual=profile('9d54e32d-18fb-5bec-62d4-1d7409204fc4')
    root=tmp_path;arm='BASE4_PLUS_YANG';seed=20261018
    baseline={'path':'baseline.json','sha256':'a'*64};budget_ref={'path':'budget.json','sha256':'b'*64}
    limit_ref={'path':'limits.json','sha256':'c'*64};code={'native.py':'d'*64}
    plan={'optimizer_updates':64,'plan_sha256':'e'*64,'output':'runs/source_extensions/'+arm+'/seed-20261018'}
    budget={'source_plans':{str(seed):plan},'source_pilot':{'path':'pilot.json'}}
    old_identity={'arm':arm,'seed':seed,'plan_sha256':plan['plan_sha256'],'budget':budget_ref,
        'limits':limit_ref,'baseline_f0':baseline,'runtime':old,'code':code}
    class Run:
        def __init__(self,*a):
            self.identity={'runtime':old};self.budget_ref=budget_ref;self.wall=types.SimpleNamespace(remaining_seconds=9999)
            self.state={'stages':{'source-f0-'+str(seed):{'kind':'source_f0','identity_sha256':digest(old_identity),'dispatched':True}}}
            self.recorded_actual=a[-1]
            runs.append(self)
        def save(self):pass
        def observe(self,*a):pass
        def guard(self):pass
        def close(self):pass
    runs=[];package=types.ModuleType('f0_probe_pkg');package.__path__=[]
    extension=types.ModuleType('f0_probe_pkg.source_extension_training')
    extension.execute_extension=lambda *a,**k:{'status':'PAUSED_NOT_COMPLETE'}
    monkeypatch.setitem(sys.modules,'f0_probe_pkg',package)
    monkeypatch.setitem(sys.modules,'f0_probe_pkg.source_extension_training',extension)
    claim=root/'runs/source_extensions/claims'/f'{arm}-seed-{seed}.json';claim.parent.mkdir(parents=True);claim.write_text('{}')
    def put(path,value):path.write_text(json.dumps(value))
    ns={'__package__':'f0_probe_pkg','DEVELOPMENT_SEEDS':(seed,), 'DownstreamError':RuntimeError,
        '_require_owned_stage':lambda:None,'Path':Path,'_latest_catalog':lambda r,c:c,
        '_primary':lambda *a:None,'_budget':lambda *a:(budget,budget_ref),'_runtime':lambda:actual,
        '_owned':lambda r,a,p:r/p,'_f0_limits':lambda *a:{'fixed':1},'immutable_json':put,
        'descriptor':lambda *a:limit_ref,'_primary_seed':lambda *a:baseline,'_code':lambda *a:code,
        'ArmRun':Run,'digest':digest,'local':lambda r,p,**k:r/p,'_source_call_floor':lambda *a:0,
        '_deadline':lambda *a:nullcontext()}
    tree=ast.parse(SOURCE.read_bytes());definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare_f0')
    definition.decorator_list=[]  # Thread-pool limits do not change this dispatch/identity branch.
    if corrected:
        for index,node in enumerate(definition.body):
            if ast.unparse(node).startswith('run = ArmRun('):
                # The semicolon places `result=None` in a distinct AST node;
                # correcting identity after either node is before dispatch.
                definition.body.insert(index+1,ast.parse('identity = _paper3_f0_lineage(root, run, identity, runtime)').body[0]);break
        else:raise AssertionError('Native ArmRun anchor absent')
        monkeypatch.syspath_prepend('/workspace/paper3-relaunch-repair')
        monkeypatch.syspath_prepend('/workspace/paper3-r1-hardware-fix')
        import runtime_transfer,audited_runtime
        journal=[]
        def validated_transfer(root,before,after,context,artifact_refs):
            assert audited_runtime.validate_typed_transfer(before,after)
            journal.append((before,after,context));return True
        monkeypatch.setattr(runtime_transfer,'validate_transfer',validated_transfer)
        ns['_paper3_f0_lineage']=boot._f0_stage_identity
        ns['_journal_for_test']=journal
    exec(compile(ast.fix_missing_locations(ast.Module(body=[definition],type_ignores=[])),str(SOURCE),'exec'),ns)
    return ns['prepare_f0'],root,{'arm':arm,'extension_manifest':{'path':'extension.json','sha256':'f'*64}},seed,runs,old_identity,actual


def test_original_native_f0_rejects_same_trajectory_after_gpu_uuid_move(tmp_path,monkeypatch):
    fn,root,catalog,seed,runs,identity,actual=fixture(tmp_path,monkeypatch)
    with pytest.raises(RuntimeError,match='source_f0_arm_dispatch_identity_changed'):
        fn(root,catalog,seed,execute=True,user_started_gpu=True)
    assert runs[0].recorded_actual==actual


def test_only_administrative_stage_runtime_needs_original_claim_lineage(tmp_path,monkeypatch):
    fn,root,catalog,seed,runs,identity,actual=fixture(tmp_path,monkeypatch,corrected=True)
    answer=fn(root,catalog,seed,execute=True,user_started_gpu=True)
    assert answer['status']=='PAUSED_NOT_COMPLETE'
    assert runs[0].state['stages']['source-f0-'+str(seed)]['identity_sha256']==digest(identity)
    assert runs[0].recorded_actual==actual


def test_f0_lineage_uses_real_audited_transfer_journal_and_preserves_claim(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('_f0_real_transfer','/workspace/paper3-relaunch-repair/runtime_transfer.py')
    transfer=importlib.util.module_from_spec(spec);spec.loader.exec_module(transfer)
    monkeypatch.setitem(sys.modules,'runtime_transfer',transfer)
    monkeypatch.syspath_prepend('/workspace/paper3-r1-hardware-fix')
    import audited_runtime
    audited_runtime.install(tmp_path,transfer_module=transfer)
    manifest=tmp_path/'payload.json';manifest.write_text(json.dumps({'files':[transfer._file(transfer.__file__)]}))
    monkeypatch.setenv('PAPER3_RUNTIME_TRANSFER_MANIFEST',str(manifest))
    old=profile('b4f600a0-a5a4-b950-504b-b8c069e3e812');actual=profile('9d54e32d-18fb-5bec-62d4-1d7409204fc4')
    identity={'arm':'BASE4_PLUS_YANG','seed':20261018,'runtime':actual,'budget':{'unchanged':True}}
    claim=tmp_path/'runs/source_extension_downstream/BASE4_PLUS_YANG/resources/claim.json'
    claim.parent.mkdir(parents=True);claim.write_text(json.dumps({'runtime':old}));before=claim.read_bytes()
    run=types.SimpleNamespace(identity={'runtime':old})
    answer=boot._f0_stage_identity(tmp_path,run,identity,actual)
    assert answer==dict(identity,runtime=old) and identity['runtime']==actual
    assert claim.read_bytes()==before
    journals=list((tmp_path/'.paper3-hpc/continuous-repair/runtime-transfers').glob('*.json'))
    assert len(journals)==1
    proof=json.loads(journals[0].read_text())
    assert proof['historical_runtime']==old and proof['actual_runtime']==actual
    assert proof['checkpoint_bytes_changed'] is False
    wrong=dict(actual,torch='2.6.0')
    with pytest.raises(ValueError,match='Software, code, data'):
        boot._f0_stage_identity(tmp_path,run,dict(identity,runtime=wrong),wrong)
    assert claim.read_bytes()==before
    assert list((tmp_path/'.paper3-hpc/continuous-repair/runtime-transfers').glob('*.json'))==journals
