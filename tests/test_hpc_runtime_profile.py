"""Installed-runtime metadata tests; no GPU operation, installation or fit.

Distribution/module observations below are explicit software doubles of the
school environment, not a claim that the local CPU executor has those versions.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hpc import runtime_profile as profile
from tools import run_controlled_pilot as pilot
from p3work import source_training as source

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    root = tmp_path / 'school project'; root.mkdir()
    (root / 'requirements-gpu.lock').write_bytes((ROOT / 'requirements-gpu.lock').read_bytes())
    for name in ('SLURM_GPUS', 'SLURM_GPUS_ON_NODE'):
        monkeypatch.delenv(name, raising=False)
    values = {'PAPER3_HPC_RUNTIME_PROFILE': profile.PROFILE,
              'PAPER3_HPC_ROOT': str(root), 'PAPER3_HPC_STAGE': 'train',
              'SLURM_JOB_ID': '777777', 'SLURMD_NODENAME': 'gpu3', 'SLURM_JOB_GPUS': '0'}
    for key, value in values.items(): monkeypatch.setenv(key, value)
    monkeypatch.setattr(profile.platform, 'python_version', lambda: '3.12.3')
    installed = dict(profile.REGISTRY['pins'])
    installed.update({'requests': '2.32.3', 'nvidia-cublas-cu12': '12.1.3.1'})
    distributions = [SimpleNamespace(metadata={'Name': name}, version=version)
                     for name, version in installed.items()]
    monkeypatch.setattr(profile.metadata, 'distributions', lambda: iter(distributions))
    # This fixture explicitly doubles the child process observation boundary.
    # Real child-process behavior is exercised by the tests below.
    monkeypatch.setattr(profile, '_observe_installed_inventory', lambda root: profile._installed_inventory())
    monkeypatch.setattr(profile, '_runtime_versions', lambda: dict(profile.REGISTRY['pins']))
    return root, distributions


def state(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}


def test_capture_complete_actual_inventory_and_immutable_registered_provenance(runtime):
    root, distributions = runtime; original_lock = (root / 'requirements-gpu.lock').read_bytes()
    descriptor = profile.prepare_runtime_profile(root, profile.PROFILE)
    actual = json.loads((root / profile.INVENTORY).read_bytes())
    expected = sorted([{'name': d.metadata['Name'], 'version': d.version} for d in distributions],
                      key=lambda row: row['name'])
    assert actual == expected and len(actual) == 7
    evidence = json.loads((root / profile.EVIDENCE).read_bytes())
    assert evidence['profile'] == 'school-cu121-v1'
    assert evidence['original_RunPod_lock_sha256'] == hashlib.sha256(original_lock).hexdigest()
    assert evidence['resolved_dependencies']['sha256'] == hashlib.sha256((root / profile.INVENTORY).read_bytes()).hexdigest()
    assert evidence['inventory_packages'] == 7 and evidence['packages_installed_by_capture'] == 0
    assert evidence['scientific_parameters_changed'] is False
    assert descriptor == profile.validate_runtime_profile(root, profile.PROFILE)
    before = state(root)
    assert profile.prepare_runtime_profile(root, profile.PROFILE) == descriptor
    assert state(root) == before and (root / 'requirements-gpu.lock').read_bytes() == original_lock


def test_node_overlay_equal_duplicate_distributions_are_canonicalized(runtime):
    root, distributions = runtime
    distributions.append(SimpleNamespace(metadata={'Name': 'TORCH'}, version='2.5.1+cu121'))
    profile.prepare_runtime_profile(root, profile.PROFILE)
    rows = json.loads((root / profile.INVENTORY).read_bytes())
    assert sum(row['name'] == 'torch' for row in rows) == 1


def test_conflicting_installed_versions_block_before_any_audit_write(runtime):
    root, distributions = runtime
    distributions.append(SimpleNamespace(metadata={'Name': 'torch'}, version='2.8.0+cu128'))
    with pytest.raises(profile.RuntimeProfileError, match='conflicting_installed'):
        profile.prepare_runtime_profile(root, profile.PROFILE)
    assert not (root / 'audit').exists()


@pytest.mark.parametrize('name,version', [('torch','2.8.0+cu128'), ('numpy','2.3.5'), ('pytest','8.4.1')])
def test_registered_core_versions_are_exact(runtime,name,version):
    root, distributions = runtime
    next(d for d in distributions if d.metadata['Name'] == name).version = version
    with pytest.raises(profile.RuntimeProfileError, match='registered_pin_mismatch'):
        profile.prepare_runtime_profile(root, profile.PROFILE)
    assert not (root / 'audit').exists()


def test_imported_modules_must_match_distribution_metadata(runtime,monkeypatch):
    root, _ = runtime
    monkeypatch.setattr(profile, '_runtime_versions', lambda: {**profile.REGISTRY['pins'], 'torch':'2.8.0+cpu'})
    with pytest.raises(profile.RuntimeProfileError, match='imported_modules_mismatch'):
        profile.prepare_runtime_profile(root, profile.PROFILE)
    assert not (root / 'audit').exists()


def test_actual_python_patch_version_is_registered(runtime,monkeypatch):
    root, _ = runtime
    monkeypatch.setattr(profile.platform, 'python_version', lambda:'3.12.14')
    with pytest.raises(profile.RuntimeProfileError, match='python_version_mismatch'):
        profile.prepare_runtime_profile(root, profile.PROFILE)


@pytest.mark.parametrize('key,value', [
    ('PAPER3_HPC_RUNTIME_PROFILE','foreign-profile'), ('PAPER3_HPC_RUNTIME_PROFILE',''),
    ('PAPER3_HPC_ROOT','/another/project'), ('PAPER3_HPC_STAGE','test'),
    ('SLURM_JOB_ID',''), ('SLURM_JOB_GPUS',''), ('SLURMD_NODENAME','login1')])
def test_explicit_profile_project_and_gpu_allocation_required_before_writing(runtime,monkeypatch,key,value):
    root, _ = runtime; monkeypatch.setenv(key,value)
    with pytest.raises(profile.RuntimeProfileError):
        profile.prepare_runtime_profile(root, profile.PROFILE)
    assert not (root / 'audit').exists()


def test_no_profile_selection_does_not_silently_select_school(runtime,monkeypatch):
    root, _ = runtime; monkeypatch.delenv('PAPER3_HPC_RUNTIME_PROFILE')
    assert profile.selected_profile(root) is None
    with pytest.raises(profile.RuntimeProfileError, match='explicit_selection'):
        profile.prepare_runtime_profile(root, profile.PROFILE)


def test_existing_different_dependency_evidence_is_never_overwritten(runtime):
    root, _ = runtime; path=root/profile.INVENTORY;path.parent.mkdir()
    path.write_text('[{"name":"torch","version":"2.8.0+cu128"}]\n')
    before=state(root)
    with pytest.raises(profile.RuntimeProfileError,match='existing_evidence_differs'):
        profile.prepare_runtime_profile(root,profile.PROFILE)
    assert state(root)==before and not (root/profile.EVIDENCE).exists()


@pytest.mark.parametrize('changed',[profile.INVENTORY,profile.EVIDENCE,'requirements-gpu.lock'])
def test_modified_inventory_provenance_or_original_lock_blocks_reuse(runtime,changed):
    root,_=runtime;profile.prepare_runtime_profile(root,profile.PROFILE)
    path=root/changed;path.write_bytes(path.read_bytes()+b'\n')
    before=state(root)
    with pytest.raises(profile.RuntimeProfileError):profile.validate_runtime_profile(root,profile.PROFILE)
    with pytest.raises(profile.RuntimeProfileError):profile.prepare_runtime_profile(root,profile.PROFILE)
    assert state(root)==before


def test_complete_inventory_rechecks_non_core_installed_packages(runtime):
    root,distributions=runtime;profile.prepare_runtime_profile(root,profile.PROFILE)
    distributions.append(SimpleNamespace(metadata={'Name':'new-package'},version='1.0'))
    with pytest.raises(profile.RuntimeProfileError,match='full_inventory_changed'):
        profile.validate_runtime_profile(root,profile.PROFILE)


@pytest.mark.parametrize('relative',[profile.INVENTORY,profile.EVIDENCE,'requirements-gpu.lock'])
def test_symlink_evidence_or_lock_is_not_accepted(runtime,relative):
    root,_=runtime;profile.prepare_runtime_profile(root,profile.PROFILE)
    path=root/relative;retained=path.with_name(path.name+'.retained')
    path.rename(retained);path.symlink_to(retained)
    with pytest.raises(profile.RuntimeProfileError,match='symlink'):
        profile.validate_runtime_profile(root,profile.PROFILE)


def test_metadata_capture_never_copies_extra_distribution_metadata(runtime):
    root,distributions=runtime;distributions[0].metadata['SECRET']='DO_NOT_COPY'
    profile.prepare_runtime_profile(root,profile.PROFILE)
    assert all(b'DO_NOT_COPY' not in value for value in state(root).values())


def test_pilot_dependency_adapter_selects_school_profile_only_explicitly(runtime,monkeypatch):
    from tools import run_controlled_pilot as pilot
    root,_=runtime;profile.prepare_runtime_profile(root,profile.PROFILE)
    descriptor=pilot._resolved_dependencies(root)
    assert descriptor['path']==profile.INVENTORY
    assert descriptor['sha256']==hashlib.sha256((root/profile.INVENTORY).read_bytes()).hexdigest()
    monkeypatch.delenv('PAPER3_HPC_RUNTIME_PROFILE')
    with pytest.raises(pilot.PilotGateError):pilot._resolved_dependencies(root)


@pytest.mark.parametrize('identity_profile,receipt_profile,selection',[
    (None,None,True), ({'path':profile.EVIDENCE,'sha256':'0'*64},None,True),
    ({'path':profile.EVIDENCE,'sha256':'0'*64},None,False)])
def test_source_verifier_requires_same_explicit_profile_in_identity_and_receipt(
        runtime,monkeypatch,identity_profile,receipt_profile,selection):
    from p3work import source_training as source
    root,_=runtime;profile.prepare_runtime_profile(root,profile.PROFILE)
    if not selection:monkeypatch.delenv('PAPER3_HPC_RUNTIME_PROFILE')
    with pytest.raises(source.SourceTrainingError,match='runtime profile'):
        source._verify_pilot_result_manifests(SimpleNamespace(root=root),root/'missing.json',
            {'deployment_runtime_profile':receipt_profile},{'deployment_runtime_profile':identity_profile},{})


@pytest.mark.parametrize('missing',['identity','receipt'])
def test_valid_profile_must_be_bound_by_both_pilot_documents(runtime,missing):
    root,_=runtime;ref=profile.prepare_runtime_profile(root,profile.PROFILE)
    identity={'deployment_runtime_profile':ref};receipt={'deployment_runtime_profile':ref}
    (identity if missing=='identity' else receipt).pop('deployment_runtime_profile')
    with pytest.raises(source.SourceTrainingError,match='runtime profile identity differs'):
        source._verify_pilot_result_manifests(SimpleNamespace(root=root),root/'missing.json',receipt,identity,{})


def test_source_manifest_verifier_binds_registered_profile_and_complete_inventory(runtime):
    """Manifest-only SOFTWARE_TEST_DOUBLE; never a completed-pilot proof."""
    root,_=runtime;ref=profile.prepare_runtime_profile(root,profile.PROFILE)
    folder=root/'runs/SOFTWARE_TEST_DOUBLE_MANIFEST';folder.mkdir(parents=True)
    def descriptor(path):
        raw=path.read_bytes()
        return {'path':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    dependency=descriptor(root/profile.INVENTORY)
    identity={'resolved_dependencies':dependency,'deployment_runtime_profile':ref,
              'torch':'2.5.1+cu121','numpy':'2.2.6','fixture_scope':'SOFTWARE_TEST_DOUBLE_MANIFEST_ONLY'}
    source.atomic_json(folder/'identity.json',identity)
    claim=root/'runs/controlled_pilot_job_claim.json'
    source.atomic_json(claim,{'scope':'ENGINEERING_SOURCE_PILOT_ONLY','max_jobs':1,
                             'output':folder.relative_to(root).as_posix()})
    predictions=[]
    for name in ('engineering_prediction_replay.json','continuous_predictions.json','resumed_predictions.json'):
        path=folder/name;source.atomic_json(path,{'fixture_scope':'SOFTWARE_TEST_DOUBLE_NO_GPU_OR_FIT'})
        predictions.append(descriptor(path))
    proof={'engineering_prediction_replay':predictions[0],'prediction_artifacts':predictions[1:]}
    receipt={'status':'ENGINEERING_PILOT_COMPLETE','checkpoint_files':{},'resolved_dependencies':dependency,
             'deployment_runtime_profile':ref,'fixture_scope':'SOFTWARE_TEST_DOUBLE_MANIFEST_ONLY'}
    paths=[claim,root/profile.INVENTORY,root/profile.EVIDENCE,folder/'identity.json',
           *(root/row['path'] for row in predictions)]
    files=[{**descriptor(path),'role':'result','dataset':'Paper3SourceEngineering'} for path in paths]
    manifest=folder/'engineering_artifact_manifest.json'
    source.atomic_json(manifest,{'schema_version':'paper3-source-engineering-artifacts-v1',
        'scope':'ENGINEERING_SOURCE_PILOT_ONLY','scientific_result':False,
        'pilot_status':'ENGINEERING_PILOT_COMPLETE','files':files})
    receipt['artifact_manifest']=descriptor(manifest)
    backup=folder/'engineering_backup_manifest.json'
    receipt['backup_manifest_path']=backup.relative_to(root).as_posix()
    receipt_file=folder/'engineering_receipt.json';source.atomic_json(receipt_file,receipt)
    source.atomic_json(backup,{'schema_version':1,'files':files+[
        {**descriptor(path),'role':'result','dataset':'Paper3SourceEngineering'} for path in (manifest,receipt_file)]})
    result=source._verify_pilot_result_manifests(SimpleNamespace(root=root),receipt_file,receipt,identity,proof)
    assert result['artifact_manifest']==descriptor(manifest)
    assert result['backup_manifest_is_upload_receipt'] is False
    (root/profile.EVIDENCE).write_bytes((root/profile.EVIDENCE).read_bytes()+b'\n')
    with pytest.raises(source.SourceTrainingError,match='provenance_changed'):
        source._verify_pilot_result_manifests(SimpleNamespace(root=root),receipt_file,receipt,identity,proof)


def test_fresh_observer_retains_exact_interpreter_environment_and_bounded_command(tmp_path,monkeypatch):
    observed = []
    monkeypatch.setenv('PAPER3_HPC_NODE_CACHE_ROOT', '/software-fixture/node-cache')
    monkeypatch.setenv('PYTHONPATH', '/software-fixture/site')
    monkeypatch.setenv('HTTPS_PROXY', 'http://SOFTWARE_PROXY_VALUE_NOT_TO_PRINT')
    inventory = [{'name':'software-fixture','version':'1.0'}]
    def runner(command, **kwargs):
        observed.append((command, kwargs))
        assert os.environ['PYTHONPATH'] == '/software-fixture/site'
        assert os.environ['PAPER3_HPC_NODE_CACHE_ROOT'] == '/software-fixture/node-cache'
        assert os.environ['HTTPS_PROXY'] == 'http://SOFTWARE_PROXY_VALUE_NOT_TO_PRINT'
        return SimpleNamespace(returncode=0,stdout=profile._raw(inventory),stderr=b'')
    assert profile._observe_installed_inventory(tmp_path,runner=runner) == inventory
    command, kwargs = observed[0]
    assert command[0:3] == [sys.executable,'-B','-c']
    assert command[-2:] == [str(Path(profile.__file__).absolute()),'--observe-installed-inventory']
    assert kwargs == {'cwd':tmp_path,'stdin':subprocess.DEVNULL,'stdout':subprocess.PIPE,
                      'stderr':subprocess.PIPE,'close_fds':True,'check':False,'timeout':60}
    assert 'env' not in kwargs and 'SOFTWARE_PROXY' not in repr(command)


@pytest.mark.parametrize('error,code',[
    (subprocess.TimeoutExpired(['observer'],60,stderr=b'SOFTWARE_SECRET'), 'observer_timeout'),
    (FileNotFoundError('SOFTWARE_SECRET'), 'observer_unavailable'),
])
def test_observer_process_failures_are_bounded_and_do_not_expose_child_output(tmp_path,error,code):
    def runner(*args,**kwargs): raise error
    with pytest.raises(profile.RuntimeProfileError,match=code) as caught:
        profile._observe_installed_inventory(tmp_path,runner=runner)
    assert 'SOFTWARE_SECRET' not in str(caught.value)


@pytest.mark.parametrize('status,raw,expected',[
    (2,profile._raw({'error':'runtime_profile_conflicting_installed_distributions'}),'conflicting_installed'),
    (2,profile._raw({'error':'SOFTWARE_SECRET'}),'observer_failed'),
    (0,b'SOFTWARE_SECRET not JSON','observer_output_invalid'),
    (0,b'[]\n','observer_output_invalid'),
    (0,b'[{"name":"fixture","version":"1"}]\n','observer_output_invalid'),
    (0,profile._raw([{'name':'fixture','version':'1','secret':'SOFTWARE_SECRET'}]),'observer_output_invalid'),
    (0,profile._raw([{'name':'fixture','version':'1'},{'name':'fixture','version':'2'}]),'observer_output_invalid'),
    (0,profile._raw([{'name':'z','version':'1'},{'name':'a','version':'1'}]),'observer_output_invalid'),
    (0,b'x'*(profile.INVENTORY_OBSERVER_MAX_BYTES+1),'observer_output_invalid'),
])
def test_observer_output_is_canonical_complete_bounded_and_safe(tmp_path,status,raw,expected):
    def runner(*args,**kwargs):
        return SimpleNamespace(returncode=status,stdout=raw,stderr=b'SOFTWARE_SECRET')
    with pytest.raises(profile.RuntimeProfileError,match=expected) as caught:
        profile._observe_installed_inventory(tmp_path,runner=runner)
    assert 'SOFTWARE_SECRET' not in str(caught.value)


def _distribution(folder,name,version):
    info=folder/(name.replace('-','_')+'-'+version+'.dist-info')
    info.mkdir(parents=True)
    (info/'METADATA').write_text('Metadata-Version: 2.1\nName: '+name+'\nVersion: '+version+'\n')
    return info


def test_real_fresh_observer_ignores_only_parent_dynamic_path_not_installed_packages(tmp_path,monkeypatch):
    """Actual stdlib metadata subprocess; no model, optimizer, CUDA or fit."""
    baseline=profile._observe_installed_inventory(tmp_path)
    assert baseline
    selected=next(row for row in baseline if row['name']=='packaging')
    vendor=tmp_path/'dynamic-vendor'
    _distribution(vendor,selected['name'],'0.0.0')
    # This reproduces setuptools' runtime sys.path.extend(_vendor) mechanism.
    monkeypatch.syspath_prepend(str(vendor))
    with pytest.raises(profile.RuntimeProfileError,match='conflicting_installed'):
        profile._installed_inventory()
    assert profile._observe_installed_inventory(tmp_path) == baseline


def test_real_fresh_observer_detects_installed_package_add_version_change_and_removal(tmp_path,monkeypatch):
    """An ordinary startup search-root change remains complete evidence."""
    baseline=profile._observe_installed_inventory(tmp_path)
    site=tmp_path/'software-fixture-site'
    info=_distribution(site,'paper3-observer-software-fixture','1.0')
    prior=os.environ.get('PYTHONPATH')
    monkeypatch.setenv('PYTHONPATH',str(site)+(os.pathsep+prior if prior else ''))
    installed=profile._observe_installed_inventory(tmp_path)
    assert installed == sorted([*baseline,{'name':'paper3-observer-software-fixture','version':'1.0'}],
                              key=lambda row:row['name'])
    (info/'METADATA').write_text('Metadata-Version: 2.1\nName: paper3-observer-software-fixture\nVersion: 1.1\n')
    changed=profile._observe_installed_inventory(tmp_path)
    assert changed != installed
    assert {'name':'paper3-observer-software-fixture','version':'1.1'} in changed
    (info/'METADATA').unlink();info.rmdir()
    assert profile._observe_installed_inventory(tmp_path) == baseline


def test_actual_setuptools_import_cannot_change_fresh_installed_inventory(tmp_path):
    script = '''
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from hpc.runtime_profile import _observe_installed_inventory, _installed_inventory, RuntimeProfileError
root=Path.cwd()
before=_observe_installed_inventory(root)
paths=list(sys.path)
import setuptools
try:
    parent_inventory=_installed_inventory()
    parent_status='changed' if parent_inventory!=before else 'unchanged'
except RuntimeProfileError as exc:
    parent_status=str(exc)
after=_observe_installed_inventory(root)
assert before==after
assert 'torch' not in sys.modules
print(json.dumps({'scope':'ACTUAL_METADATA_IMPORT_NO_CUDA_FIT_OR_INSTALL',
                  'fresh_inventory_unchanged':True,'parent_status':parent_status,
                  'dynamic_path_added':len(sys.path)>len(paths)}))
'''
    result=subprocess.run([sys.executable,'-B','-c',script,str(ROOT)],cwd=tmp_path,
                          stdin=subprocess.DEVNULL,capture_output=True,check=False,timeout=15)
    assert result.returncode==0, result.stderr.decode(errors='replace')
    observation=json.loads(result.stdout)
    assert observation['fresh_inventory_unchanged'] is True
    assert observation['scope']=='ACTUAL_METADATA_IMPORT_NO_CUDA_FIT_OR_INSTALL'
