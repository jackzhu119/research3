import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def setup(root):
    driver=load('observed_driver_compat',HERE/'r1_driver_compat.py')
    audited=load('original_audited_runtime',Path('/workspace/paper3-r1-hardware-fix/audited_runtime.py'))
    transfer=load('original_runtime_transfer',Path('/workspace/paper3-relaunch-repair/runtime_transfer.py'))
    transfer._manifest=lambda:{'path':'existing-payload.json','sha256':'a'*64,'bytes':123}
    driver.install(root,audited_module=audited)
    audited.install(root,transfer_module=transfer)
    return driver,audited,transfer


def profiles(reverse=False):
    before={'python':'3.12.3','torch':'2.5.1+cu121','cuda':'12.1','cudnn':90100,
        'driver_versions':['550.54.14']*8,'device_name':'NVIDIA A100-SXM4-80GB',
        'device_uuid':'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
        'device_total_memory_bytes':80*1024**3,'device_compute_capability':[8,0],
        'autocast':False,'tf32':False,'float32_matmul_precision':'highest',
        'deterministic_algorithms':True,'cublas_workspace_config':':4096:8',
        'code':{'p3work/meta_training.py':'1'*64,'p3work/contracts.py':'2'*64}}
    actual=copy.deepcopy(before)
    actual.update(driver_versions=['535.104.12']*2,device_name='NVIDIA A100-PCIE-40GB',
        device_uuid='GPU-ffffffff-1111-2222-3333-444444444444',device_total_memory_bytes=40*1024**3)
    return (actual,before) if reverse else (before,actual)


@pytest.mark.parametrize('reverse',[False,True])
def test_only_observed_compatible_driver_pair_allowed_with_actual_profiles_retained(tmp_path,reverse):
    driver,audited,transfer=setup(tmp_path)
    before,actual=profiles(reverse);saved=copy.deepcopy((before,actual))
    policy=audited.validate_typed_transfer(before,actual)
    assert 'driver_versions' in policy['allowed_differences']
    assert policy['driver_transition']['historical_driver_versions']==before['driver_versions']
    assert policy['driver_transition']['actual_driver_versions']==actual['driver_versions']
    assert policy['driver_transition']['bitwise_equivalence_or_equal_speed_claimed'] is False
    assert (before,actual)==saved


def test_original_p_engineering_meta_gate_now_accepts_pair_and_journals_exact_observations(tmp_path):
    driver,audited,transfer=setup(tmp_path)
    before,actual=profiles();proof={'runtime_profile':before,'job':{'method':'P','dimension':8}}
    saved=copy.deepcopy((proof,actual))
    assert transfer.meta_gate(tmp_path,proof,actual,SimpleNamespace(identity='P8-seed20261018')) is True
    journal=json.loads((tmp_path/audited.AUDIT).read_text())
    record=next(iter(journal['transfers'].values()))
    assert record['historical_runtime']==before and record['actual_runtime']==actual
    assert record['context']=='meta_engineering_timing:P8-seed20261018'
    assert record['policy']['driver_transition']['observed_driver_pair_only'] is True
    assert (proof,actual)==saved


@pytest.mark.parametrize('seed',[20261018,20261019,20261020])
def test_original_whole_meta_identity_reconcile_preserves_checkpoint_and_scientific_keys(tmp_path,seed):
    driver,audited,transfer=setup(tmp_path)
    before,actual=profiles()
    shared={'job':{'method':'P','dimension':8,'radius':0.05,'seed':seed,'phase':'screening' if seed==20261018 else 'confirmation'},
        'budget_constraints':{'stage':'meta_training','max_outer_updates':1000,'max_jobs':1,
                              'engineering_receipt':{'path':'existing/receipt.json','sha256':'3'*64}},
        'episode_plan_sha256':'4'*64,'source_model_state_sha256':'5'*64,'signal_contract_sha256':'6'*64,
        'stieger_manifest_sha256':'7'*64,'full_unroll':True}
    historical={**before,**copy.deepcopy(shared)};candidate={**actual,**copy.deepcopy(shared)}
    folder=tmp_path/'runs/meta/existing-job';folder.mkdir(parents=True)
    (folder/'identity.json').write_text(json.dumps(historical))
    (folder/'latest_checkpoint.json').write_text(json.dumps({'path':'outer-0350.pt','outer_cursor':350}))
    (folder/'outer-0350.pt').write_bytes(b'original optimizer/RNG/sampler/tensor checkpoint bytes\x00')
    existing={p:p.read_bytes() for p in folder.iterdir()}
    answer=transfer.reconcile_identity(tmp_path,folder,candidate,'meta')
    assert answer==historical
    assert candidate=={**actual,**shared}
    assert all(path.read_bytes()==value for path,value in existing.items())
    journal=json.loads((tmp_path/audited.AUDIT).read_text())
    record=next(iter(journal['transfers'].values()))
    assert record['actual_runtime']==candidate and record['historical_runtime']==historical


def test_source_extension_full_python_build_identity_reconcile_preserves_checkpoint(tmp_path):
    driver,audited,transfer=setup(tmp_path)
    before,actual=profiles()
    python_build='3.12.3 (main, Apr 10 2024, 17:19:04) [GCC 11.4.0]'
    for profile in (before,actual):
        profile['python']=python_build
        profile['cuda_runtime']=profile.pop('cuda')
        profile['device_total_memory']=profile.pop('device_total_memory_bytes')
        profile['compute_capability']=profile.pop('device_compute_capability')
    shared={'job':{'dataset':'extension','seed':20261020},'plan_sha256':'f'*64,
            'budget_sha256':'e'*64,'scientific_inputs_sha256':'d'*64}
    historical={**copy.deepcopy(shared),'runtime':before}
    candidate={**copy.deepcopy(shared),'runtime':actual}
    folder=tmp_path/'runs/source_extension/existing-fit';folder.mkdir(parents=True)
    (folder/'identity.json').write_text(json.dumps(historical))
    (folder/'last_state.pt').write_bytes(b'unchanged extension optimizer RNG sampler checkpoint')
    existing={path:path.read_bytes() for path in folder.iterdir()}
    assert transfer.reconcile_identity(tmp_path,folder,candidate,'source')==historical
    assert candidate=={**shared,'runtime':actual}
    assert all(path.read_bytes()==value for path,value in existing.items())
    record=next(iter(json.loads((tmp_path/audited.AUDIT).read_text())['transfers'].values()))
    assert record['historical_runtime']==before and record['actual_runtime']==actual
    assert record['actual_runtime']['python']==python_build


def test_unequal_full_python_build_strings_remain_rejected(tmp_path):
    driver,audited,transfer=setup(tmp_path)
    before,actual=profiles()
    before['python']='3.12.3 (main, Apr 10 2024) [GCC 11.4.0]'
    actual['python']='3.12.3 (main, Apr 11 2024) [GCC 11.4.0]'
    with pytest.raises(ValueError):audited.validate_typed_transfer(before,actual)


@pytest.mark.parametrize('key,value',[
    ('python','3.12.4'),('torch','2.6.0+cu121'),('cuda','12.4'),('cudnn',90200),
    ('autocast',True),('tf32',True),('float32_matmul_precision','high'),
    ('deterministic_algorithms',False),('cublas_workspace_config',':16:8'),
    ('code',{'p3work/meta_training.py':'changed-source'}),
    ('source_model_state_sha256','changed-data'),('max_outer_updates',999),
    ('job',{'method':'P','dimension':4,'seed':20261019}),
])
def test_non_driver_software_code_data_numerical_or_scientific_changes_still_rejected(tmp_path,key,value):
    driver,audited,transfer=setup(tmp_path)
    before,actual=profiles()
    before.update(source_model_state_sha256='original-data',max_outer_updates=1000,
                  job={'method':'P','dimension':8,'seed':20261018})
    actual.update(source_model_state_sha256='original-data',max_outer_updates=1000,
                  job={'method':'P','dimension':8,'seed':20261018})
    actual[key]=value;saved=copy.deepcopy((before,actual))
    with pytest.raises(ValueError):audited.validate_typed_transfer(before,actual)
    assert (before,actual)==saved


@pytest.mark.parametrize('drivers',[['535.104.13'],['550.54.15'],['525.60.12'],['535.104.12','550.54.14'],[],None])
def test_unknown_mixed_or_below_minimum_driver_versions_are_not_normalized(tmp_path,drivers):
    driver,audited,transfer=setup(tmp_path)
    before,actual=profiles();actual['driver_versions']=drivers
    with pytest.raises(ValueError):audited.validate_typed_transfer(before,actual)


def test_existing_same_driver_inventory_count_normalization_is_preserved(tmp_path):
    driver,audited,transfer=setup(tmp_path)
    before,actual=profiles();before['driver_versions']=['535.104.12']*8
    policy=audited.validate_typed_transfer(before,actual)
    assert 'driver_transition' not in policy
    assert policy['identical_driver_versions_with_different_node_GPU_inventory_count'] is True


def test_install_is_idempotent_and_native_source_bytes_are_not_modified(tmp_path):
    driver=load('driver_idempotence',HERE/'r1_driver_compat.py')
    audited_path=Path('/workspace/paper3-r1-hardware-fix/audited_runtime.py');original=audited_path.read_bytes()
    audited=load('audited_for_idempotence',audited_path)
    assert driver.install(tmp_path,audited_module=audited)['status']=='INSTALLED'
    wrapped=audited.validate_typed_transfer
    assert driver.install(tmp_path,audited_module=audited)['status']=='ALREADY_INSTALLED'
    assert audited.validate_typed_transfer is wrapped
    assert audited_path.read_bytes()==original


def test_reinstall_for_different_root_rejects_conflicting_binding(tmp_path):
    driver,audited,transfer=setup(tmp_path/'original-root')
    wrapped=audited.validate_typed_transfer
    with pytest.raises(ValueError,match='different project root'):
        driver.install(tmp_path/'different-root',audited_module=audited)
    assert audited.validate_typed_transfer is wrapped


def test_existing_audited_install_closure_sees_later_helper_and_preserves_code_rejection(tmp_path):
    driver=load('driver_after_audited_install',HERE/'r1_driver_compat.py')
    audited=load('audited_install_first',Path('/workspace/paper3-r1-hardware-fix/audited_runtime.py'))
    native_calls=[]
    def original_uuid_validator(*args,**kwargs):
        native_calls.append((args,kwargs))
        raise AssertionError('Changed raw driver versions must use the journaled observed-pair path')
    transfer=SimpleNamespace(validate_transfer=original_uuid_validator,
        _manifest=lambda:{'path':'original-payload.json','sha256':'9'*64,'bytes':100})
    # This is the production installer order: the existing audited installer
    # binds its closure first, then the new helper amends its global comparator.
    audited.install(tmp_path,transfer_module=transfer)
    driver.install(tmp_path,audited_module=audited)
    before,actual=profiles();preserved=copy.deepcopy((before,actual))
    assert transfer.validate_transfer(tmp_path,before,actual,'future_meta_confirmation') is True
    assert (before,actual)==preserved
    changed=copy.deepcopy(actual);changed['code']['p3work/contracts.py']='changed-source'
    with pytest.raises(ValueError):
        transfer.validate_transfer(tmp_path,before,changed,'future_meta_confirmation_changed_code')
    assert not native_calls
