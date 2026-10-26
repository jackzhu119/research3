import copy
import json
from pathlib import Path
import sys

import pytest

from paper3r10.platform import common, runtime

PACKAGE=Path(__file__).resolve().parents[1]


def plan_fixture(tmp_path,tasks=None):
    root=tmp_path/'project';root.mkdir(exist_ok=True)
    output=root/'runs/round10-test'
    plan={'schema_version':common.PLAN_SCHEMA,'round':10,'stage':'mechanism','synthetic_fixture':True,
          'recipe_sha256':'a'*64,'root':str(root),'output':str(output),'package':str(PACKAGE),
          'code_manifest':{'paper3r10/platform/common.py':common.sha(PACKAGE/'paper3r10/platform/common.py')},
          'python':sys.executable,'secondary_ablation_family':[],'reference_strategy':'fixture-fixed','tasks':tasks or [dict(key='mechanism-a',kind='mechanism',role='meta_validation',
                     config={'method':'fixture'},input_identities={'data':'b'*64},requires=[])]}
    registry=root/'registry.json';common.atomic(registry,{'fixture':True})
    plan['study_registry']={'path':'registry.json','sha256':common.sha(registry),'bytes':registry.stat().st_size}
    plan=common.sign_plan(plan)
    path=tmp_path/'plan.json';common.atomic(path,plan)
    return path,plan


def write_plan(path,plan):
    signed=common.sign_plan(plan);plan.clear();plan.update(signed);common.atomic(path,plan)


def complete(plan,task,*,paper=True):
    folder=common.task_folder(plan,task);folder.mkdir(parents=True,exist_ok=True)
    p=folder/'participants.csv';p.write_bytes(b'participant_id,ba\nS01,0.5\n')
    item={'path':p.name,'sha256':common.sha(p),'bytes':p.stat().st_size}
    result={'status':'COMPLETE','recipe_sha256':plan['recipe_sha256'],'task_key':task['key'],
            'task_identity':common.task_identity(plan,task),'artifacts':[item],
            'paper_artifacts':[item] if paper else [],'synthetic':True,
            'scientific_scope':'SOFTWARE_TEST_ONLY' if plan.get('synthetic_fixture') else 'REAL_DATA'}
    common.atomic(folder/'result.json',result)
    return common.seal_result(plan,task)


def test_prepare_is_metadata_only_and_never_imports_torch_or_calls_slurm(tmp_path,monkeypatch):
    path,plan=plan_fixture(tmp_path)
    def forbidden(*a,**k):raise AssertionError('No external command in prepare')
    monkeypatch.setattr(runtime.subprocess,'run',forbidden)
    monkeypatch.setattr(runtime.subprocess,'Popen',forbidden)
    before=set(sys.modules)
    prepared=runtime.start(path,submit=False)
    assert prepared['status']=='PREPARED_NOT_SUBMITTED'
    assert prepared['submitted_jobs']==0
    assert 'torch' not in set(sys.modules)-before
    assert (Path(plan['output'])/'plan.json').read_bytes()==path.read_bytes()
    assert not (runtime.runtime_dir(plan)/'SUBMIT_AUTHORIZATION.json').exists()


@pytest.mark.parametrize('change',[
    {'schema_version':'round9'},{'round':9},{'recipe_sha256':'not-a-sha'},
    {'stage':'unregistered'},
    {'tasks':[dict(key='../escape',role='meta_validation')]},
    {'tasks':[dict(key='x',role='internal_holdout')]},
    {'tasks':[dict(key='x',role='meta_validation',requires=['missing'])]},
    {'tasks':[dict(key='x',role='meta_validation',requires=['y']),dict(key='y',role='meta_train',requires=['x'])]},
])
def test_invalid_or_leaking_plan_rejected(tmp_path,change):
    path,plan=plan_fixture(tmp_path);plan.update(change);common.atomic(path,plan)
    with pytest.raises(ValueError):common.load_plan(path)


def test_formal_stage_requires_exact_development_only_freeze_and_unused_confirmation(tmp_path,monkeypatch):
    import paper3r10.confirmation
    monkeypatch.setattr(paper3r10.confirmation,'validate_confirmation_metadata',lambda *_: {'contract_fixture':True})
    path,plan=plan_fixture(tmp_path);plan['stage']='formal_exploratory';plan['synthetic_fixture']=False;write_plan(path,plan)
    with pytest.raises(ValueError,match='freeze'):common.load_plan(path)
    freeze=Path(plan['root'])/'freeze.json'
    common.atomic(freeze,dict(status='FROZEN',code_manifest_sha256=plan['code_manifest_sha256'],
        study_registry_sha256=plan['study_registry']['sha256'],scientific_scope='REAL_DATA',secondary_ablation_family=[],reference_strategy='fixture-fixed',selection_roles=['meta_train','meta_validation'],
        target_query_labels_used_for_selection=False,development_query_labels_used=True,
        selected_configuration={'strength':1},comparison_family=['new-vs-EA']))
    plan['freeze']={'path':freeze.name,'sha256':common.sha(freeze),'bytes':freeze.stat().st_size}
    plan['comparison_family']=['new-vs-EA'];plan['tasks'][0].update(config={'strength':1},is_primary_algorithm=True)
    write_plan(path,plan);common.load_plan(path)
    plan['stage']='formal_confirmatory';write_plan(path,plan)
    with pytest.raises(ValueError,match='confirmation'):common.load_plan(path)
    manifest=Path(plan['root'])/'new-cohort.json'
    common.atomic(manifest,dict(cohort_kind='independent_confirmation',previously_used_for_algorithm_design=False,
                               dataset_identity='new-real-data-sha',participants=['new:S1']))
    plan['confirmation_manifest']={'path':manifest.name,'sha256':common.sha(manifest),'bytes':manifest.stat().st_size}
    write_plan(path,plan);common.load_plan(path)
    plan['previously_used_participants']=['new:S1'];write_plan(path,plan)
    with pytest.raises(ValueError,match='overlap'):common.load_plan(path)


def test_exact_sealed_completion_and_changed_payload_or_recipe_refuses_reuse(tmp_path):
    _,plan=plan_fixture(tmp_path);task=plan['tasks'][0]
    complete(plan,task);assert common.verify_result(plan,task)
    (common.task_folder(plan,task)/'participants.csv').write_text('modified')
    assert not common.verify_result(plan,task)
    assert common.verify_result(plan,task,verify_artifacts=False)  # explicit fast monitor semantics
    changed=copy.deepcopy(plan);changed['recipe_sha256']='c'*64
    assert not common.verify_result(changed,task)


def test_result_or_artifact_path_escape_and_missing_seal_not_completed(tmp_path):
    _,plan=plan_fixture(tmp_path);task=plan['tasks'][0];folder=common.task_folder(plan,task)
    complete(plan,task);(folder/'completion.json').unlink()
    assert not common.verify_result(plan,task)
    result=common.read(folder/'result.json');result['artifacts'][0]['path']='../elsewhere'
    common.atomic(folder/'result.json',result)
    with pytest.raises(ValueError):common.seal_result(plan,task)


def test_task_identity_includes_configuration_seed_and_input_manifest(tmp_path):
    _,plan=plan_fixture(tmp_path);task=plan['tasks'][0];old=common.task_identity(plan,task)
    for key,value in [('config',{'different':1}),('input_identities',{'data':'changed'}),('seed',42)]:
        new=dict(task,**{key:value});assert common.task_identity(plan,new)!=old


def test_controller_cannot_query_or_submit_without_explicit_authorization(tmp_path):
    path,_=plan_fixture(tmp_path);calls=[]
    with pytest.raises(ValueError,match='explicit --submit'):
        runtime.controller_loop(path,runner=lambda *a,**k:calls.append(a),once=True)
    assert not calls


@pytest.mark.parametrize('field,value',[
    ('config',{'strength':999}),('input_identities',{'data':'new-but-not-resigned'}),
    ('baseline_artifacts',{'old':{'path':'new-checkpoint.pt','sha256':'d'*64,'bytes':100}}),
])
def test_task_changes_cannot_keep_the_old_scientific_recipe(tmp_path,field,value):
    path,plan=plan_fixture(tmp_path)
    plan['tasks'][0][field]=value;common.atomic(path,plan)
    with pytest.raises(ValueError,match='without a new recipe'):common.load_plan(path)


def test_declared_code_change_is_detected_even_after_plan_resigning(tmp_path):
    path,plan=plan_fixture(tmp_path)
    package=tmp_path/'round10-package';package.mkdir()
    (package/'algorithm.py').write_text('value = 1\n')
    plan.update(package=str(package),code_manifest={'algorithm.py':common.sha(package/'algorithm.py')})
    write_plan(path,plan);common.load_plan(path)
    (package/'algorithm.py').write_text('value = 2\n')
    with pytest.raises(ValueError,match='code changed'):common.load_plan(path)
    plan['tasks'][0]['seed']=123;write_plan(path,plan)
    with pytest.raises(ValueError,match='code changed'):common.load_plan(path)


def test_formal_primary_config_ablations_and_comparisons_match_freeze(tmp_path):
    path,plan=plan_fixture(tmp_path);plan['stage']='formal_exploratory';plan['synthetic_fixture']=False
    freeze=Path(plan['root'])/'freeze.json'
    frozen=dict(status='FROZEN',code_manifest_sha256=plan['code_manifest_sha256'],
        study_registry_sha256=plan['study_registry']['sha256'],scientific_scope='REAL_DATA',secondary_ablation_family=[],reference_strategy='fixture-fixed',selection_roles=['meta_train','meta_validation'],
        target_query_labels_used_for_selection=False,development_query_labels_used=True,
        selected_configuration={'strength':1},comparison_family=['new-vs-EA'],
        allowed_ablations={'remove-certificate':{'strength':1,'certificate':False}})
    common.atomic(freeze,frozen)
    plan.update(freeze={'path':freeze.name,'sha256':common.sha(freeze),'bytes':freeze.stat().st_size},
                comparison_family=['new-vs-EA'])
    plan['tasks'][0].update(config={'strength':1},is_primary_algorithm=True)
    write_plan(path,plan);common.load_plan(path)
    plan['tasks'][0]['config']['strength']=2;write_plan(path,plan)
    with pytest.raises(ValueError,match='Primary'):common.load_plan(path)
    plan['tasks'][0]['config']['strength']=1
    plan['tasks'].append(dict(key='ablation',kind='evaluate',config={'strength':1,'certificate':False},
                              ablation='remove-certificate',method='remove-certificate',role='exploratory',requires=[]))
    write_plan(path,plan);common.load_plan(path)
    plan['tasks'][1]['config']['strength']=.5;write_plan(path,plan)
    with pytest.raises(ValueError,match='ablation'):common.load_plan(path)
    plan['tasks'].pop();plan['comparison_family']=['new-vs-weak-control'];write_plan(path,plan)
    with pytest.raises(ValueError,match='comparison family'):common.load_plan(path)


def test_child_slot_ordinal_and_determinism_environment(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','7')
    monkeypatch.setenv('PYTHONPATH','untrusted')
    env=runtime.child_environment(2,'claim-token','cuda:0')
    assert env['CUDA_VISIBLE_DEVICES']=='7' and env['PAPER3_R10_SLOT']=='2'
    assert env['CUBLAS_WORKSPACE_CONFIG']==':4096:8' and 'PYTHONPATH' not in env
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES');monkeypatch.delenv('SLURM_JOB_GPUS',raising=False)
    monkeypatch.delenv('SLURM_STEP_GPUS',raising=False)
    with pytest.raises(ValueError):runtime.child_environment(1,'x','cuda:0')
