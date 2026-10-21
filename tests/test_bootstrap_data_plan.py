"""Pure software fixture checks; no real data, network, SDK install or fit."""
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import io
import json
import os
import shutil
import tarfile

import pytest

from tools import bootstrap_data_plan as b


def put(root,path,data):
    target=root/path;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(data if isinstance(data,bytes) else (json.dumps(data)+'\n').encode())
    return b.file_ref(root,target)


def receipt(rows,profile):
    value={'schema_version':1,'operation':'paper3_explicit_manifest_backup','status':'verified','execute':True,
        'verification':'full_get_body_sha256','prefix':'paper3/bootstrap-fixture/','objects':[]}
    if profile=='source_extension':
        from p3work import source_extension_io
        value['source_extension_profile']=source_extension_io.profile()
    for row in rows:
        row={k:row[k] for k in b.ENTRY_FIELDS}
        value['objects'].append({**row,'key':value['prefix']+row['path'],'status':'verified',
            'put':{'status':'created','http_status':200,'request_body_bytes':row['bytes'],'request_body_sha256':row['sha256']},
            'get':{'status':'verified','http_status':200,'complete_body_read':True,'bytes_read':row['bytes'],
                'sha256':row['sha256'],'declared_content_length':row['bytes']}})
    return value


@pytest.fixture
def package(tmp_path,monkeypatch):
    root=tmp_path/'research3';root.mkdir()
    original=Path(b.__file__).resolve().parents[1]
    for name in ['tools/r2_backup.py','tools/r2_restore.py','p3work/source_extension_io.py']:
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original/name,target)
    # A tiny, explicitly fixture-bound runtime avoids using the real archive.
    runtime=put(root,b.RUNTIME,b'fixture-runtime')
    monkeypatch.setattr(b,'RUNTIME_SHA',runtime['sha256']);monkeypatch.setattr(b,'RUNTIME_BYTES',runtime['bytes'])
    rows=[{**runtime,'role':'archive','dataset':'PythonRuntime312','io_profile':'baseline'}]
    for name in b.SDK_NAMES:
        item=put(root,f'data/dependency_wheels/cpu/{name}-1-py3-none-any.whl',name.encode())
        rows.append({**item,'role':'archive','dataset':'Dependencies','io_profile':'baseline'})
    requirements=put(root,'requirements-gpu.lock',b'fixture==1\n')
    contract=put(root,'contracts/portable_python_312.json',{'runtime_archive':runtime})
    for item in [requirements,contract]:rows.append({**item,'role':'protocol','dataset':'Paper3','io_profile':'baseline'})
    for name in ['Yang2025','Dreyer2023']:
        item=put(root,f'cache/source_extensions_20261019/{name}/fixture/source_cache.json',{'fixture':True})
        rows.append({**item,'role':'result','dataset':name,'io_profile':'source_extension'})
    batches={profile:{'schema_version':1,'files':[{k:r[k] for k in b.ENTRY_FIELDS} for r in rows if r['io_profile']==profile]} for profile in ['baseline','source_extension']}
    manifest=put(root,'audit/deployment/deploy_manifest.json',{'schema_version':'paper3-data-plan-cache-only-deploy-v1',
        'status':'FIXED_LOCAL_CACHE_ONLY_BACKUP_MANIFEST_NOT_YET_UPLOADED','closure_sha256':'c'*64,
        'raw_payloads_selected':0,'backup_manifests':batches})
    receipts=[]
    for profile in batches:
        item=put(root,f'audit/deployment/{profile}_receipt.json',receipt([r for r in rows if r['io_profile']==profile],profile))
        receipts.append(root/item['path'])
    index=b.freeze_index(root,root/manifest['path'],receipts)
    index_path=root/'audit/deployment/bootstrap_index.json'
    b.write_once(index_path,index)
    return root,index,index_path,receipts


def rewrite_index(index_path,index):
    index['index_sha256']=b.digest({k:v for k,v in index.items() if k!='index_sha256'})
    index_path.write_text(json.dumps(index))


def test_freeze_actual_full_get_receipts_cover_every_file(package):
    root,index,path,_=package
    assert b.validate_index(root,path)==index
    assert len(index['bootstrap_wheels'])==8
    assert len(index['restore_groups'])==2
    assert sum(len(g['paths']) for g in index['restore_groups'])==len(index['files'])
    assert all(not p.startswith('data/source') for g in index['restore_groups'] for p in g['paths'])


def test_dry_run_has_no_subprocess_or_write(package):
    root,index,path,_=package
    def forbidden(*args,**kwargs):raise AssertionError('dry run must not spawn')
    result=b.prepare(root,path,runner=forbidden)
    assert result['network_operations']==result['scientific_fits']==0
    assert result['status']=='DRY_RUN_FIXED_INDEX_AND_EMBEDDED_FILES_VERIFIED'
    assert not (root/b.STATE).exists()
    assert b.RUNTIME in result['embedded_files']


def test_missing_real_full_get_coverage_cannot_freeze(package):
    root,index,_,receipts=package
    with pytest.raises(b.BootstrapError,match='every_deployment_file'):
        b.freeze_index(root,root/index['deploy_manifest']['path'],receipts[:1])


def test_receipt_partial_get_is_rejected(package):
    root,index,_,receipts=package
    value=json.loads(receipts[0].read_bytes());value['objects'][0]['get']['complete_body_read']=False
    receipts[0].write_text(json.dumps(value))
    with pytest.raises(Exception):b.freeze_index(root,root/index['deploy_manifest']['path'],receipts)


@pytest.mark.parametrize('path',['../oops','/tmp/data','a//b','a/./b','a/../b','a\\b','a\x00b','.env','a/.aws/key','paper1/cache.json','a/secrets.json','p3work/private_key.pem'])
def test_unsafe_paths_rejected(tmp_path,path):
    with pytest.raises(b.BootstrapError):b.checked_path(tmp_path,path)


@pytest.mark.parametrize('path',['data/raw/a.edf','data/dependency_wheels/wrong/a.whl','cache/source_v2/a.npz','runs/fit.json','cache/foreign/a.npy','audit/raw.edf','runtime/arbitrary.tar.gz'])
def test_raw_or_legacy_cannot_enter_payload(path):
    row={'path':path,'role':'audit','dataset':'Paper3','sha256':'a'*64,'bytes':1}
    with pytest.raises(b.BootstrapError):b.safe_payload(row)


def test_symlink_components_rejected(tmp_path):
    (tmp_path/'outside').mkdir();(tmp_path/'link').symlink_to(tmp_path/'outside')
    with pytest.raises(b.BootstrapError,match='symlink'):b.checked_path(tmp_path,'link/file.json')


def test_runtime_tamper_stops_before_subprocess(package):
    root,_,path,_=package;(root/b.RUNTIME).write_bytes(b'changed')
    with pytest.raises(b.BootstrapError,match='SHA_or_size'):b.prepare(root,path)
    assert not (root/b.STATE).exists()


def test_index_tamper_is_rejected(package):
    root,index,path,_=package;index['raw_payloads_selected']=1;path.write_text(json.dumps(index))
    with pytest.raises(b.BootstrapError,match='identity_or_scope'):b.validate_index(root,path)


def test_restore_selection_duplicate_is_rejected_even_with_new_digest(package):
    root,index,path,_=package;index['restore_groups'].append(copy.deepcopy(index['restore_groups'][0]));rewrite_index(path,index)
    with pytest.raises(b.BootstrapError,match='scope_mismatch'):b.validate_index(root,path)


def test_profile_cannot_change_for_selected_file(package):
    root,index,path,_=package;index['restore_groups'][0]['io_profile']='source_extension';rewrite_index(path,index)
    with pytest.raises(b.BootstrapError,match='scope_mismatch'):b.validate_index(root,path)


def test_embedded_SDK_cannot_be_missing(package):
    root,index,path,_=package;index['bootstrap_wheels'].pop();rewrite_index(path,index)
    with pytest.raises(b.BootstrapError,match='closed_bootstrap_SDK'):b.validate_index(root,path)


@pytest.mark.parametrize('execute,user_started',[(False,False),(False,True),(True,False)])
def test_training_requires_both_explicit_flags(package,execute,user_started):
    root,_,path,_=package
    with pytest.raises(b.BootstrapError,match='launch_requires'):b.prepare(root,path,execute=execute,launch=True,user_started_gpu=user_started)
    assert not (root/b.STATE).exists()


def test_thread_environment_keeps_secrets_unmodified_and_pins_threads(monkeypatch):
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY','fixture-not-a-real-key')
    monkeypatch.setenv('OMP_NUM_THREADS','99');env=b.thread_environment()
    assert all(env[k]=='1' for k in b.THREAD_VARIABLES)
    assert env['AWS_SECRET_ACCESS_KEY']==os.environ['AWS_SECRET_ACCESS_KEY']
    assert os.environ['OMP_NUM_THREADS']=='99'
    assert env['PIP_CONFIG_FILE']==os.devnull and env['PIP_NO_INDEX']=='1'


def test_write_once_does_not_replace_other_identity(tmp_path):
    p=tmp_path/'proof.json';b.write_once(p,{'a':1});b.write_once(p,{'a':1})
    with pytest.raises(b.BootstrapError):b.write_once(p,{'a':2})
    assert b.read(p)=={'a':1}


def test_subprocess_uses_array_and_only_owned_log(tmp_path):
    calls=[]
    def runner(args,**kwargs):calls.append((args,kwargs));return SimpleNamespace(returncode=0)
    b.run_command(['python','file with spaces.py','--no-index'],root=tmp_path,env={},log=tmp_path/'logs/run.log',runner=runner)
    assert calls[0][0]==['python','file with spaces.py','--no-index']
    assert 'shell' not in calls[0][1]


def test_subprocess_failure_suppresses_error_content(tmp_path):
    with pytest.raises(b.BootstrapError,match='local_subprocess_failed'):
        b.run_command(['bad'],root=tmp_path,env={},log=tmp_path/'run.log',runner=lambda *a,**k:SimpleNamespace(returncode=2))


def test_disk_gate_runs_before_network_and_allows_completed_environment_resumption(package,monkeypatch):
    root,index,_,_=package
    monkeypatch.setattr(shutil,'disk_usage',lambda p:SimpleNamespace(free=2*1024**3))
    with pytest.raises(b.BootstrapError,match='insufficient_free_disk'):b.disk_preflight(root,index)
    put(root,b.STATE+'/offline_environment_ready.json',{'index_sha256':index['index_sha256']})
    assert b.disk_preflight(root,index)['required_free_bytes']==1024**3


def test_existing_payload_conflict_not_counted_as_resumable(package):
    root,index,_,_=package
    (root/index['files'][0]['path']).write_bytes(b'wrong')
    with pytest.raises(b.BootstrapError,match='SHA_or_size'):b.disk_preflight(root,index)


def make_tar(root,members):
    data=io.BytesIO()
    with tarfile.open(fileobj=data,mode='w:gz') as archive:
        for name,payload,kind in members:
            item=tarfile.TarInfo(name)
            if kind=='file':item.size=len(payload);archive.addfile(item,io.BytesIO(payload))
            elif kind=='symlink':item.type=tarfile.SYMTYPE;item.linkname=payload;archive.addfile(item)
    return put(root,b.RUNTIME,data.getvalue())


def test_runtime_extract_preserves_internal_links_and_owned_resume(tmp_path):
    runtime=make_tar(tmp_path,[('python/bin/python3.12',b'fixture','file'),('python/bin/python3','python3.12','symlink')])
    index={'runtime':runtime};python=b.extract_runtime(tmp_path,index)
    assert python.read_bytes()==b'fixture' and python.is_symlink()
    assert b.extract_runtime(tmp_path,index)==python


@pytest.mark.parametrize('member',[('../escape',b'x','file'),('/tmp/escape',b'x','file'),('other/file',b'x','file'),('python/bin/python3','../../../escape','symlink')])
def test_runtime_tar_escape_rejected_without_writing_escape(tmp_path,member):
    runtime=make_tar(tmp_path,[member])
    with pytest.raises(b.BootstrapError,match='unsafe_portable|link_escapes'):b.extract_runtime(tmp_path,{'runtime':runtime})
    assert not (tmp_path/'escape').exists()


def test_unowned_runtime_not_deleted(tmp_path):
    target=tmp_path/b.STATE/'python-runtime';target.mkdir(parents=True);(target/'keep').write_text('owned by someone else')
    with pytest.raises(b.BootstrapError,match='not_owned'):b.extract_runtime(tmp_path,{'runtime':{}})
    assert (target/'keep').is_file()


def test_launch_orders_main_before_two_separate_source_arms(package):
    root,index,_,_=package
    put(root,b.STATE+'/offline_environment_ready.json',{'index_sha256':index['index_sha256']})
    calls=[]
    def runner(args,**kwargs):calls.append((args,kwargs['env']));return SimpleNamespace(returncode=0)
    result=b.launch_training(root,index,Path('/fixture/python'),user_started_gpu=True,runner=runner)
    assert len(calls)==5 and 'tools/run_training_campaign.py' in calls[0][0]
    assert 'BASE4_PLUS_YANG' in calls[1][0] and 'BASE4_PLUS_YANG' in calls[2][0]
    assert 'BASE4_PLUS_DREYER' in calls[3][0] and 'BASE4_PLUS_DREYER' in calls[4][0]
    assert all('--execute' in command and '--user-started-gpu' in command for command,_ in calls)
    assert all(all(env[k]=='1' for k in b.THREAD_VARIABLES) for _,env in calls)
    assert result['server_actions'] is False and result['external_query_accessed'] is False


def test_failed_main_stops_before_any_extension(package):
    root,index,_,_=package;put(root,b.STATE+'/offline_environment_ready.json',{'index_sha256':index['index_sha256']})
    calls=[]
    def runner(args,**kwargs):calls.append(args);return SimpleNamespace(returncode=3)
    with pytest.raises(b.BootstrapError):b.launch_training(root,index,Path('/fixture/python'),user_started_gpu=True,runner=runner)
    assert len(calls)==1


def test_extension_snapshot_is_finite_partial_results_not_completion(tmp_path):
    put(tmp_path,'runs/source_extensions/BASE4_PLUS_YANG/seed-1/fit/selected.pt',b'fixture')
    put(tmp_path,'runs/source_extension_downstream/BASE4_PLUS_DREYER/controller/claim.json',{'partial':True})
    put(tmp_path,'runs/source_extensions/claims/BASE4_PLUS_YANG-seed-1.json',{'calls':4})
    put(tmp_path,'runs/unrelated/weights.pt',b'not selected')
    dry=b.snapshot_extension_results(tmp_path)
    assert dry['file_count']==3 and not (tmp_path/dry['manifest_path']).exists()
    actual=b.snapshot_extension_results(tmp_path,execute=True)
    snapshot=b.read(tmp_path/actual['manifest_path'])
    assert actual['fit_claims_deleted'] is False
    assert snapshot['scientific_completion_claimed'] is False
    assert all(row['path'].startswith(('runs/source_extensions/','runs/source_extension_downstream/')) for row in snapshot['files'])
    assert b.snapshot_extension_results(tmp_path,execute=True)['manifest']==actual['manifest']


def test_snapshot_rejects_symlink_or_private_file_without_copying(tmp_path):
    root=tmp_path/'runs/source_extensions/BASE4_PLUS_YANG';root.mkdir(parents=True)
    (root/'secrets.json').write_text('fixture')
    with pytest.raises(b.BootstrapError,match='private_or_foreign'):b.snapshot_extension_results(tmp_path,execute=True)
    assert not (tmp_path/'audit').exists()


def test_snapshot_cannot_run_while_owned_stage_locked(tmp_path):
    import fcntl
    root=tmp_path/'runs/source_extension_downstream/BASE4_PLUS_YANG/controller';root.mkdir(parents=True)
    (root/'claim.json').write_text('{}')
    with (root/'owner.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(b.BootstrapError,match='stop_owned_extension_workers'):
            b.snapshot_extension_results(tmp_path,execute=True)
    assert not (tmp_path/'audit').exists()


def test_empty_extension_snapshot_is_not_fabricated(tmp_path):
    with pytest.raises(b.BootstrapError,match='no_actual_owned'):b.snapshot_extension_results(tmp_path,execute=True)


def test_restore_dispatches_exact_profile_paths_and_reverifies_all_files(package,monkeypatch):
    from tools import r2_restore
    from p3work import source_extension_io
    root,index,_,_=package;calls=[]
    def fake_restore(**kwargs):
        calls.append(kwargs)
        result={'status':'restored_verified','objects':[{'path':p,'status':'already_present_verified'} for p in kwargs['paths']]}
        kwargs['report_path'].write_text(json.dumps(result))
        return result
    monkeypatch.setattr(r2_restore,'run_restore',fake_restore);monkeypatch.setattr(source_extension_io,'run_restore',fake_restore)
    report=b.restore_files(root,index)
    assert report['restored_this_invocation']==0
    assert report['already_present_verified']==len(index['files'])
    assert len(calls)==2
    for call,group in zip(calls,index['restore_groups']):
        assert call['paths']==group['paths'] and call['execute'] is True
        assert 'all_approved' not in call
    assert report['GPU_checked'] is False


def test_failed_restore_stops_before_later_groups(package,monkeypatch):
    from tools import r2_restore
    root,index,_,_=package;calls=[]
    def fake_restore(**kwargs):calls.append(kwargs);return {'status':'failed','objects':[]}
    monkeypatch.setattr(r2_restore,'run_restore',fake_restore)
    with pytest.raises(b.BootstrapError,match='restore_not_complete'):b.restore_files(root,index)
    assert len(calls)==1
    assert not (root/b.STATE/'restore_complete').exists()


def test_two_restore_workers_cannot_duplicate_downloads(package):
    import fcntl
    root,index,_,_=package;parent=root/b.STATE;parent.mkdir()
    with (parent/'restore_worker.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(b.BootstrapError,match='restore_worker_already_active'):b.restore_files(root,index)


def test_full_dependency_install_is_pure_offline_and_strips_recorded_index_url(package):
    root,index,_,_=package
    data='--extra-index-url https://fixture.invalid/do-not-contact\n'+'\n'.join(f'fixture{i}==1' for i in range(100))+'\n'
    item=put(root,'requirements-gpu.lock',data.encode());index['requirements']=item
    for row in index['files']:
        if row['path']=='requirements-gpu.lock':row.update(item)
    calls=[]
    def runner(args,**kwargs):calls.append(args);return SimpleNamespace(returncode=0)
    b.install_full_profile(root,index,Path('/fixture/python'),runner=runner)
    assert len(calls)==3 and '--no-index' in calls[0]
    assert '--extra-index-url' not in (root/b.STATE/'requirements-offline.lock').read_text()
    assert b.read(root/b.STATE/'offline_environment_ready.json')['GPU_checked'] is False


def test_invalid_cache_SHA_not_accepted_by_recomputed_index(package):
    root,index,path,_=package;index['files'][0]['sha256']='not-a-digest';rewrite_index(path,index)
    with pytest.raises(b.BootstrapError,match='SHA_or_size'):b.validate_index(root,path)


def test_runtime_directory_symlink_not_treated_as_owned(tmp_path):
    external=tmp_path/'outside';(external/'bin').mkdir(parents=True);(external/'bin/python3').write_text('fixture')
    parent=tmp_path/b.STATE;parent.mkdir();(parent/'python-runtime').symlink_to(external,target_is_directory=True)
    b.write_once(parent/'runtime_identity.json',{'schema_version':b.SCHEMA,'archive':{},'root':str(tmp_path)})
    with pytest.raises(b.BootstrapError,match='directory_symlink'):b.extract_runtime(tmp_path,{'runtime':{}})


def test_owned_runtime_interpreter_cannot_resolve_to_host(tmp_path):
    target=tmp_path/b.STATE/'python-runtime'
    (target/'bin').mkdir(parents=True)
    outside=tmp_path/'host-python';outside.write_text('fixture')
    (target/'bin/python3').symlink_to(outside)
    b.write_once(tmp_path/b.STATE/'runtime_identity.json',
        {'schema_version':b.SCHEMA,'archive':{},'root':str(tmp_path)})
    with pytest.raises(b.BootstrapError,match='inside_owned_runtime'):
        b.extract_runtime(tmp_path,{'runtime':{}})


def test_owned_runtime_preserves_official_internal_interpreter_link(tmp_path):
    target=tmp_path/b.STATE/'python-runtime'
    (target/'bin').mkdir(parents=True)
    (target/'bin/python3.12').write_text('fixture')
    (target/'bin/python3').symlink_to('python3.12')
    b.write_once(tmp_path/b.STATE/'runtime_identity.json',
        {'schema_version':b.SCHEMA,'archive':{},'root':str(tmp_path)})
    assert b.extract_runtime(tmp_path,{'runtime':{}})==target/'bin/python3'


def test_rerun_retains_first_real_get_proof_in_unique_attempt(package,monkeypatch):
    from tools import r2_restore
    from p3work import source_extension_io
    root,index,_,_=package;invocations=[]
    def fake_restore(**kwargs):
        number=len(invocations)//2
        result={'status':'restored_verified','objects':[{'path':p,'status':'restored_verified' if number==0 else 'already_present_verified',
            'complete_body_read':number==0} for p in kwargs['paths']]}
        kwargs['report_path'].write_text(json.dumps(result));invocations.append(kwargs['report_path']);return result
    monkeypatch.setattr(r2_restore,'run_restore',fake_restore);monkeypatch.setattr(source_extension_io,'run_restore',fake_restore)
    first=b.restore_files(root,index);before=[b.file_ref(root,p) for p in invocations]
    second=b.restore_files(root,index)
    assert first['attempt']!=second['attempt']
    assert first['restored_this_invocation']==len(index['files'])
    assert second['already_present_verified']==len(index['files'])
    assert all(b.file_ref(root,root/ref['path'])==ref for ref in before)
    assert len(list((root/b.STATE/'restore_complete').glob('*.json')))==2


def test_restore_failure_keeps_its_attempt_for_inspection(package,monkeypatch):
    from tools import r2_restore
    root,index,_,_=package
    def fake_restore(**kwargs):
        result={'status':'failed','objects':[]};kwargs['report_path'].write_text(json.dumps(result));return result
    monkeypatch.setattr(r2_restore,'run_restore',fake_restore)
    with pytest.raises(b.BootstrapError):b.restore_files(root,index)
    attempts=list((root/b.STATE/'restore_receipts').glob('attempt-*'))
    assert len(attempts)==1 and (attempts[0]/'invocation.json').is_file()
    assert b.read(attempts[0]/'00000.json')['status']=='failed'


def test_write_once_does_not_follow_existing_symlink(tmp_path):
    outside=tmp_path/'outside.json';outside.write_text('{}')
    target=tmp_path/'proof.json';target.symlink_to(outside)
    with pytest.raises(b.BootstrapError,match='write_symlink'):b.write_once(target,{})
    assert outside.read_text()=='{}'


def test_log_does_not_follow_symlink_outside_state_tree(tmp_path):
    outside=tmp_path/'outside.log';outside.write_text('original')
    target=tmp_path/'logs';target.symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(b.BootstrapError,match='write_symlink'):
        b.run_command(['never'],root=tmp_path,env={},log=target/'outside.log',runner=lambda *a,**k:pytest.fail('must not spawn'))
    assert outside.read_text()=='original'


def test_absolute_index_symlink_is_rejected(package):
    root,_,path,_=package;alias=root/'audit/alias.json';alias.symlink_to(path)
    with pytest.raises(b.BootstrapError,match='symlink'):b.prepare(root,alias)
    with pytest.raises(b.BootstrapError,match='symlink'):b.validate_index(root,alias)


def test_full_profile_never_overwrites_requirements_symlink(package):
    root,index,_,_=package
    data='\n'.join(f'fixture{i}==1' for i in range(100))+'\n'
    item=put(root,'requirements-gpu.lock',data.encode());index['requirements']=item
    for row in index['files']:
        if row['path']=='requirements-gpu.lock':row.update(item)
    parent=root/b.STATE;parent.mkdir();outside=root/'outside.txt';outside.write_text('preserve')
    (parent/'requirements-offline.lock').symlink_to(outside)
    with pytest.raises(b.BootstrapError,match='write_symlink'):
        b.install_full_profile(root,index,Path('/fixture/python'),runner=lambda *a,**k:pytest.fail('must not spawn'))
    assert outside.read_text()=='preserve'
