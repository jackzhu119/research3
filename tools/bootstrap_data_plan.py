#!/usr/bin/env python3
"""Restore a frozen cache-only deployment, then optionally run local GPU stages.

Stage zero uses only Python's standard library. Credentials are inherited by
existing R2 IO; they are never inspected, copied, or written by this program.
Default is read-only. --execute prepares software/data; scientific execution
also needs --launch-training --user-started-gpu after the user's manual boot.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'paper3-cache-only-bootstrap-v1'
STATE = '.paper3-deployment'
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
RUNTIME = 'runtime/cpython-3.12.15+20261013-x86_64-unknown-linux-gnu-install_only.tar.gz'
RUNTIME_SHA = '5d546fc004fc2435fa142eb8f42ed8de5eb4725963dc22f7e6671014a80eb560'
RUNTIME_BYTES = 66983838
SDK_NAMES = ('boto3', 'botocore', 's3transfer', 'jmespath', 'urllib3', 'python_dateutil', 'six', 'certifi')
ARMS = ('BASE4_PLUS_YANG', 'BASE4_PLUS_DREYER')
ENTRY_FIELDS = ('path', 'sha256', 'bytes', 'role', 'dataset')


class BootstrapError(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def checked_path(root, value, *, exists=False):
    if not isinstance(value, str) or not value or '\\' in value or any(ord(c) < 32 for c in value):
        raise BootstrapError('unsafe_relative_deployment_path')
    p = PurePosixPath(value)
    if p.is_absolute() or any(c in {'', '.', '..'} for c in value.split('/')):
        raise BootstrapError('unsafe_relative_deployment_path')
    target = Path(root)
    for component in p.parts:
        tokens = set(re.split('[^a-z0-9]+', component.lower()))
        normalized=re.sub('[^a-z0-9]+','_',component.lower())
        if (component.lower().startswith('.env') or component.lower() in {'.git', '.ssh', '.aws', '.boto', 'paper1', 'paper2'}
                or tokens & {'credential', 'credentials', 'secret', 'secrets', 'password', 'token', 'apikey', 'privatekey'}
                or any(marker in normalized for marker in ('aws_access_key','aws_secret_access_key','api_key','private_key','r2_endpoint','r2_bucket'))
                or Path(component).suffix.lower() in {'.pem','.key','.p12','.pfx'}):
            raise BootstrapError('private_or_foreign_project_path_forbidden')
        target = target/component
        if target.is_symlink():
            raise BootstrapError('deployment_symlink_forbidden')
    if exists and not target.is_file():
        raise BootstrapError('required_package_file_missing')
    return target


def file_ref(root, path):
    path = Path(path)
    path = path if path.is_absolute() else checked_path(root, str(path), exists=True)
    relative = path.relative_to(root).as_posix()
    checked_path(root, relative, exists=True)
    before=path.stat()
    sha = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        while chunk := stream.read(1024*1024):
            sha.update(chunk); size += len(chunk)
    after=path.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
        raise BootstrapError('file_changed_during_SHA_snapshot')
    return {'path': relative, 'sha256': sha.hexdigest(), 'bytes': size}


def verify_file(root, item):
    if not isinstance(item, dict) or not {'path', 'sha256', 'bytes'} <= set(item):
        raise BootstrapError('file_descriptor_required')
    if (not re.fullmatch('[a-f0-9]{64}', str(item['sha256']))
            or type(item['bytes']) is not int or item['bytes'] < 0):
        raise BootstrapError('invalid_SHA_or_size_descriptor')
    actual = file_ref(root, checked_path(root, item['path'], exists=True))
    if any(actual[k] != item[k] for k in actual):
        raise BootstrapError('local_file_SHA_or_size_mismatch')
    return actual


def safe_payload(item):
    path = item['path']; checked_path(Path('/'), path)
    p = PurePosixPath(path)
    if (item.get('role') == 'source_development_raw' or p.suffix.lower() in {'.npz', '.pt', '.pth', '.ckpt', '.pkl', '.pickle'}
            or p.parts[0] in {'runs', 'checkpoints', 'predictions'}):
        raise BootstrapError('raw_legacy_or_scientific_output_forbidden')
    if p.parts[0] == 'data':
        if p.parts[:2] != ('data', 'dependency_wheels') or len(p.parts) != 4 or p.parts[2] not in {'cpu', 'gpu'} or p.suffix != '.whl':
            raise BootstrapError('raw_payload_forbidden')
    elif p.parts[0] == 'runtime':
        if (path != RUNTIME or item['sha256'] != RUNTIME_SHA or item['bytes'] != RUNTIME_BYTES):
            raise BootstrapError('only_pinned_software_runtime_allowed')
    elif p.parts[0] == 'cache':
        if len(p.parts) < 3 or p.parts[1] not in {'source_v2', 'stieger_v2', 'source_extensions_20261019', 'source_c3czc4_v1'} or p.suffix.lower() not in {'.npy', '.json', '.jsonl', '.tsv', '.csv', '.txt'}:
            raise BootstrapError('cache_outside_frozen_namespaces')
    elif p.parts[0] not in {'audit', 'catalog', 'data_catalog', 'manifests', 'contracts', 'plans', 'protocol', 'status', 'provenance', 'p3work', 'tools', 'tests'} and path not in {'requirements-cpu.lock', 'requirements-gpu.lock', 'pyproject.toml', 'README.md'}:
        raise BootstrapError('deployment_payload_outside_allowlist')
    if p.suffix.lower() in {'.edf', '.bdf', '.gdf', '.mat', '.set', '.fdt', '.cnt', '.eeg'} and p.parts[0] != 'protocol':
        raise BootstrapError('raw_payload_forbidden')


def read(path):
    return json.loads(Path(path).read_bytes())


def no_symlink_path(path):
    absolute=Path(os.path.abspath(path));current=Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current=current/component
        if current.is_symlink():raise BootstrapError('deployment_write_symlink_forbidden')
    return absolute


def write_once(path, value):
    path = no_symlink_path(path)
    if path.exists():
        if read(path) != value: raise BootstrapError('existing_deployment_identity_must_not_change')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.bootstrap-', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally: temporary.unlink(missing_ok=True)


def load_module(root, relative, name):
    spec = importlib.util.spec_from_file_location(name, checked_path(root, relative, exists=True))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def freeze_index(root, manifest_path, receipts):
    """Local-only finalizer, after the root coordinator actually backed up data.

    The final manifest must be complete. Each selected file must be covered by
    an actual verified full-GET receipt, using its own unchanged IO policy.
    """
    root = Path(root).resolve(strict=True)
    backup = load_module(root, 'tools/r2_backup.py', '_bootstrap_backup')
    restore = load_module(root, 'tools/r2_restore.py', '_bootstrap_restore')
    extension = load_module(root, 'p3work/source_extension_io.py', '_bootstrap_extension')
    value = read(manifest_path)
    if (value.get('schema_version') != 'paper3-data-plan-cache-only-deploy-v1'
            or value.get('status') != 'FIXED_LOCAL_CACHE_ONLY_BACKUP_MANIFEST_NOT_YET_UPLOADED'
            or value.get('raw_payloads_selected') != 0):
        raise BootstrapError('actual_complete_fixed_deploy_manifest_required')
    files = []
    for profile in ('baseline', 'source_extension'):
        for row in value['backup_manifests'][profile]['files']:
            row = {**row, 'io_profile': profile}; safe_payload(row); files.append(row)
    by_path = {r['path']: r for r in files}
    if len(by_path) != len(files) or not files:
        raise BootstrapError('unique_nonempty_deployment_files_required')
    coverage, groups = set(), []
    for receipt_path in receipts:
        descriptor = file_ref(root, receipt_path)
        receipt = read(root/descriptor['path'])
        profile = 'source_extension' if 'source_extension_profile' in receipt else 'baseline'
        loader = extension._modules()[1] if profile == 'source_extension' else restore
        rows, sha = loader.load_verified_receipt(root/descriptor['path'])
        if sha != descriptor['sha256']: raise BootstrapError('receipt_changed_during_freeze')
        selected = []
        for row in rows:
            expected = by_path.get(row['path'])
            if expected is None or row['path'] in coverage: continue
            if expected['io_profile'] != profile or any(row[k] != expected[k] for k in ENTRY_FIELDS):
                raise BootstrapError('receipt_profile_or_file_identity_mismatch')
            safe_payload(row); selected.append(row['path']); coverage.add(row['path'])
        if selected: groups.append({'io_profile': profile, 'receipt': descriptor, 'paths': sorted(selected)})
    if coverage != set(by_path): raise BootstrapError('every_deployment_file_needs_real_full_GET_coverage')
    contract = read(root/'contracts/portable_python_312.json')
    if contract['runtime_archive'] != {'path':RUNTIME,'sha256':RUNTIME_SHA,'bytes':RUNTIME_BYTES}:
        raise BootstrapError('portable_runtime_contract_mismatch')
    sdk = []
    for name in SDK_NAMES:
        candidates = [r for r in files if r['path'].startswith('data/dependency_wheels/cpu/'+name+'-')]
        if len(candidates) != 1: raise BootstrapError('exact_offline_bootstrap_SDK_wheel_required')
        sdk.append({k:candidates[0][k] for k in ('path','sha256','bytes')})
    extensions = {}
    for arm, dataset in zip(ARMS, ('Yang2025','Dreyer2023')):
        matches = [r['path'] for r in files if r['path'].startswith('cache/source_extensions_20261019/'+dataset+'/') and r['path'].endswith('/source_cache.json')]
        if len(matches) != 1: raise BootstrapError('one_complete_current_source_manifest_per_arm_required')
        extensions[arm] = matches[0]
    result = {'schema_version':SCHEMA,'closure_sha256':value['closure_sha256'],
        'deploy_manifest':file_ref(root,manifest_path),'files':sorted(files,key=lambda r:r['path']),
        'restore_groups':groups,'runtime':contract['runtime_archive'],
        'runtime_contract':file_ref(root,'contracts/portable_python_312.json'),
        'requirements':file_ref(root,'requirements-gpu.lock'),'bootstrap_wheels':sdk,
        'extension_manifests':extensions,'raw_payloads_selected':0,'external_query_access_allowed':False,
        'scientific_fits_already_completed':0,'user_manual_GPU_boot_required':True,
        'thread_environment':{key:'1' for key in THREAD_VARIABLES}}
    result['index_sha256'] = digest(result)
    return result


def validate_index(root, index_path):
    root = Path(root).resolve(strict=True)
    index_path=Path(index_path)
    if index_path.is_absolute():
        try:index_path=index_path.relative_to(root)
        except ValueError:raise BootstrapError('bootstrap_index_must_be_inside_fixed_package') from None
    index_path=checked_path(root,index_path.as_posix(),exists=True)
    value = read(index_path)
    if (value.get('schema_version') != SCHEMA or digest({k:v for k,v in value.items() if k!='index_sha256'}) != value.get('index_sha256')
            or value.get('raw_payloads_selected') != 0 or value.get('external_query_access_allowed') is not False
            or value.get('user_manual_GPU_boot_required') is not True):
        raise BootstrapError('fixed_bootstrap_index_identity_or_scope_invalid')
    runtime = {'path':RUNTIME,'sha256':RUNTIME_SHA,'bytes':RUNTIME_BYTES}
    if value.get('runtime') != runtime or value.get('thread_environment') != {key:'1' for key in THREAD_VARIABLES}:
        raise BootstrapError('pinned_runtime_and_thread_profile_required')
    files = value.get('files',[]); by_path = {}
    for row in files:
        if set(row) != set(ENTRY_FIELDS)|{'io_profile'} or row['io_profile'] not in {'baseline','source_extension'}:
            raise BootstrapError('exact_profiled_file_row_required')
        if not re.fullmatch('[a-f0-9]{64}',str(row['sha256'])) or type(row['bytes']) is not int or row['bytes']<0:
            raise BootstrapError('invalid_SHA_or_size_descriptor')
        safe_payload(row)
        if row['path'] in by_path: raise BootstrapError('duplicate_deployment_file')
        by_path[row['path']] = row
    if not files: raise BootstrapError('nonempty_deployment_required')
    selected = set()
    for group in value.get('restore_groups',[]):
        verify_file(root,group['receipt'])
        if group['io_profile'] not in {'baseline','source_extension'} or not group['paths'] or len(set(group['paths']))!=len(group['paths']):
            raise BootstrapError('exact_restore_group_required')
        for path in group['paths']:
            if path not in by_path or by_path[path]['io_profile'] != group['io_profile'] or path in selected:
                raise BootstrapError('restore_group_scope_mismatch')
            selected.add(path)
    if selected != set(by_path): raise BootstrapError('all_files_must_have_one_exact_restore_selection')
    for key in ('runtime','runtime_contract','requirements','deploy_manifest'):
        verify_file(root,value[key])
    names=[]
    for row in value.get('bootstrap_wheels',[]):
        verify_file(root,row)
        if row['path'] not in by_path or any(row[k]!=by_path[row['path']][k] for k in row):
            raise BootstrapError('SDK_wheel_outside_frozen_file_set')
        names.append(PurePosixPath(row['path']).name.split('-')[0])
    if sorted(names) != sorted(SDK_NAMES): raise BootstrapError('exact_closed_bootstrap_SDK_required')
    if set(value.get('extension_manifests',{})) != set(ARMS) or any(p not in by_path for p in value['extension_manifests'].values()):
        raise BootstrapError('both_separate_source_arm_manifests_required')
    return value


def embedded_files(index, index_path):
    """Exact minimum package payload, in addition to frozen source/metadata."""
    values = [index['runtime'], index['runtime_contract'], index['requirements'], index['deploy_manifest'],
              *index['bootstrap_wheels'], *(g['receipt'] for g in index['restore_groups'])]
    return sorted({x['path'] for x in values} | {str(index_path)})


def run_command(command, *, root, env, log, runner=subprocess.run):
    log = no_symlink_path(log); log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as stream:
        result = runner([str(x) for x in command],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=False)
    if result.returncode:
        raise BootstrapError('local_subprocess_failed_see_owned_log')


def thread_environment():
    environment = os.environ.copy()
    for key in THREAD_VARIABLES: environment[key]='1'
    # All pip operations also pass --no-index. These disable user pip config
    # and accidental resolver/download access inherited from the host.
    environment['PIP_CONFIG_FILE']=os.devnull
    environment['PIP_NO_INDEX']='1'
    environment['PIP_DISABLE_PIP_VERSION_CHECK']='1'
    return environment


def owned_interpreter(target):
    interpreter=target/'bin/python3'
    if (not interpreter.is_file()
            or not interpreter.resolve(strict=True).is_relative_to(target.resolve(strict=True))):
        raise BootstrapError('portable_interpreter_must_stay_inside_owned_runtime')
    return interpreter


def extract_runtime(root,index):
    parent=checked_path(root,STATE);parent.mkdir(exist_ok=True)
    target=parent/'python-runtime'; marker=no_symlink_path(parent/'runtime_identity.json')
    identity={'schema_version':SCHEMA,'archive':index['runtime'],'root':str(root)}
    if target.is_symlink():raise BootstrapError('portable_runtime_directory_symlink_forbidden')
    if target.exists():
        if not marker.is_file() or read(marker)!=identity or not (target/'bin/python3').is_file():
            raise BootstrapError('existing_portable_runtime_not_owned_or_incomplete')
        return owned_interpreter(target)
    staging=parent/'python-runtime.partial'; claim=parent/'runtime_extract_claim.json'
    write_once(claim,identity)
    if staging.exists():
        if staging.is_symlink() or not staging.is_dir():raise BootstrapError('owned_runtime_staging_invalid')
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        with tarfile.open(root/RUNTIME,'r:gz') as package:
            members=package.getmembers()
            if not members:raise BootstrapError('empty_runtime_archive')
            for member in members:
                p=PurePosixPath(member.name)
                if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0]!='python' or not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                    raise BootstrapError('unsafe_portable_runtime_archive_member')
                if member.issym() or member.islnk():
                    link=PurePosixPath(member.linkname)
                    resolved=(staging/Path(member.name).parent/str(link)).resolve() if member.issym() else (staging/str(link)).resolve()
                    if link.is_absolute() or not resolved.is_relative_to(staging.resolve()/'python'):
                        raise BootstrapError('portable_runtime_link_escapes_archive')
            # The exact archive has a full verified SHA; paths/links were
            # independently confined above. Preserve licenses and native links.
            if hasattr(tarfile,'data_filter'):package.extractall(staging,filter='data')
            else:package.extractall(staging)
        if not (staging/'python/bin/python3').is_file():raise BootstrapError('portable_interpreter_missing')
        write_once(marker,identity)
        (staging/'python').rename(target)
        staging.rmdir()
    except Exception:
        raise
    return owned_interpreter(target)


def disk_preflight(root,index):
    missing=0
    for row in index['files']:
        target=checked_path(root,row['path'])
        if target.exists():verify_file(root,row)
        else:missing+=row['bytes']
    # The verified full dependency profile needs about 7GiB unpacked. Reserve
    # 8GiB plus 1GiB extraction/temporary headroom before any network body.
    ready=no_symlink_path(root/STATE/'offline_environment_ready.json')
    environment_ready=ready.is_file() and read(ready).get('index_sha256')==index['index_sha256']
    required=missing+(1 if environment_ready else 9)*1024**3
    if shutil.disk_usage(root).free < required:
        raise BootstrapError('insufficient_free_disk_for_cache_wheels_and_offline_environment')
    return {'missing_payload_bytes':missing,'required_free_bytes':required}


def install_sdk(root,index,*,runner=subprocess.run):
    interpreter=extract_runtime(root,index); parent=root/STATE
    identity={'schema_version':SCHEMA,'root':str(root),'index_sha256':index['index_sha256']}
    write_once(parent/'bootstrap_identity.json',identity)
    env=thread_environment();venv=parent/'venv';python=venv/'bin/python'
    if venv.is_symlink():raise BootstrapError('bootstrap_venv_directory_symlink_forbidden')
    if not python.exists():
        claim=parent/'venv_creation_identity.json'
        if venv.exists():
            if not claim.is_file() or read(claim)!=identity or venv.is_symlink():
                raise BootstrapError('unrecognized_partial_bootstrap_venv')
            shutil.rmtree(venv)
        write_once(claim,identity)
        run_command([interpreter,'-m','venv',venv],root=root,env=env,log=parent/'sdk_install.log',runner=runner)
    no_symlink_path(venv/'bin')
    if python.resolve(strict=True)!=interpreter.resolve(strict=True):
        raise BootstrapError('venv_interpreter_must_resolve_to_owned_portable_python')
    # A hard stop during venv's ensurepip can leave a usable interpreter but
    # no pip. Reapply its bundled offline bootstrap before identical SDK pins.
    run_command([python,'-m','ensurepip','--upgrade'],root=root,env=env,log=parent/'sdk_install.log',runner=runner)
    run_command([python,'-m','pip','install','--no-index','--no-deps',*[root/r['path'] for r in index['bootstrap_wheels']]],
        root=root,env=env,log=parent/'sdk_install.log',runner=runner)
    run_command([python,'-c','import boto3,botocore,s3transfer,jmespath,urllib3,dateutil,six,certifi'],
        root=root,env=env,log=parent/'sdk_install.log',runner=runner)
    return python


def _restore_files_owned(root,index):
    """Executed in the local pinned interpreter after the minimal SDK install."""
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    from tools import r2_restore
    from p3work import source_extension_io as extension
    restored=present=0
    attempt=checked_path(root,STATE+'/restore_receipts/attempt-'+uuid.uuid4().hex)
    attempt.mkdir(parents=True,exist_ok=False)
    write_once(attempt/'invocation.json',{'schema_version':SCHEMA,'index_sha256':index['index_sha256'],
        'root':str(root),'operation':'exact_cache_only_restore','groups':len(index['restore_groups'])})
    group_proofs=[]
    for number,group in enumerate(index['restore_groups']):
        loader=extension if group['io_profile']=='source_extension' else r2_restore
        report_path=attempt/f'{number:05d}.json'
        result=loader.run_restore(backup_receipt_path=root/group['receipt']['path'],root=root,
            report_path=report_path,paths=group['paths'],execute=True)
        if result.get('status')!='restored_verified':raise BootstrapError('cache_only_restore_not_complete')
        group_proofs.append(file_ref(root,report_path))
        for row in result['objects']:
            if row['status']=='restored_verified':restored+=1
            elif row['status']=='already_present_verified':present+=1
            else:raise BootstrapError('restore_object_not_verified')
    for item in index['files']:verify_file(root,item)
    result={'schema_version':SCHEMA,'status':'ALL_FROZEN_FILES_LOCALLY_SHA_VERIFIED',
        'index_sha256':index['index_sha256'],'restored_this_invocation':restored,
        'already_present_verified':present,'file_count':len(index['files']),
        'attempt':attempt.relative_to(root).as_posix(),'group_restore_receipts':group_proofs,
        'raw_payloads_selected':0,'scientific_fits':0,'GPU_checked':False}
    # Snapshot each successful restore invocation, never overwrite prior proof.
    output=root/STATE/'restore_complete'/(attempt.name+'.json')
    write_once(output,result)
    return result


def restore_files(root,index):
    parent=checked_path(root,STATE);parent.mkdir(exist_ok=True)
    owner=checked_path(root,STATE+'/restore_worker.lock').open('a')
    try:
        try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise BootstrapError('one_cache_restore_worker_already_active') from None
        return _restore_files_owned(root,index)
    finally:owner.close()


def install_full_profile(root,index,python,*,runner=subprocess.run):
    env=thread_environment();parent=root/STATE
    for item in index['files']:verify_file(root,item)
    lines=(root/index['requirements']['path']).read_text().splitlines()
    pins=[line for line in lines if line.strip() and not line.lstrip().startswith(('#','--'))]
    if len(pins)!=100 or any(not re.fullmatch(r'[A-Za-z0-9_.-]+==[^\s;]+',line) for line in pins):
        raise BootstrapError('exact_100_offline_GPU_dependency_pins_required')
    requirements=no_symlink_path(parent/'requirements-offline.lock')
    write_once(parent/'dependency_identity.json',{'index_sha256':index['index_sha256'],'requirements':index['requirements']})
    if requirements.exists() and requirements.read_text()!='\n'.join(pins)+'\n':raise BootstrapError('offline_requirements_changed')
    requirements.write_text('\n'.join(pins)+'\n')
    run_command([python,'-m','pip','install','--no-index','--find-links',root/'data/dependency_wheels/cpu',
        '--find-links',root/'data/dependency_wheels/gpu','-r',requirements],root=root,env=env,log=parent/'offline_install.log',runner=runner)
    run_command([python,'-m','pip','check'],root=root,env=env,log=parent/'offline_install.log',runner=runner)
    run_command([python,'-c','import sys,torch,numpy,scipy,mne,sklearn,boto3; assert sys.version_info[:2]==(3,12)'],
        root=root,env=env,log=parent/'offline_install.log',runner=runner)
    write_once(parent/'offline_environment_ready.json',{'schema_version':SCHEMA,'index_sha256':index['index_sha256'],
        'status':'OFFLINE_DEPENDENCIES_INSTALLED_IMPORTS_PASSED','GPU_checked':False,'scientific_fits':0})


def launch_training(root,index,python,*,user_started_gpu=False,runner=subprocess.run):
    if not user_started_gpu:raise BootstrapError('later_user_manual_GPU_boot_flag_required')
    ready=no_symlink_path(root/STATE/'offline_environment_ready.json')
    if not ready.is_file():raise BootstrapError('complete_offline_environment_required')
    if read(ready).get('index_sha256')!=index['index_sha256']:
        raise BootstrapError('offline_environment_belongs_to_different_fixed_deployment')
    for item in index['files']:verify_file(root,item)
    env=thread_environment(); parent=root/STATE
    run_command([python,'tools/run_training_campaign.py','--execute','--user-started-gpu'],
        root=root,env=env,log=parent/'main_campaign.log',runner=runner)
    # The extension initializer verifies the real all14/three-seed development
    # freeze. Mainline must finish before either independent arm is dispatched.
    freeze='runs/training_campaign_v1/pre_internal_freeze.json'
    for arm in ARMS:
        run_command([python,'tools/run_source_extension_downstream.py','--arm',arm,'--stage','initialize',
            '--primary-freeze',freeze,'--extension-manifest',index['extension_manifests'][arm],
            '--execute','--user-started-gpu'],root=root,env=env,log=parent/(arm+'.log'),runner=runner)
        run_command([python,'tools/run_source_extension_downstream.py','--arm',arm,'--stage','auto-development',
            '--execute','--user-started-gpu'],root=root,env=env,log=parent/(arm+'.log'),runner=runner)
    return {'status':'LOCAL_MAINLINE_AND_TWO_INDEPENDENT_SOURCE_ARMS_RETURNED_SUCCESS',
        'server_actions':False,'external_query_accessed':False,'scientific_results':'inspect_producer_receipts'}


def snapshot_extension_results(root,*,execute=False):
    """Finite local result snapshot after stopping the two owned arm workers.

    This performs no upload or fit and claims no scientific completion. A
    failed/partial owned run may be preserved without deleting any fit claim.
    """
    root=Path(root).resolve(strict=True);candidates=[];locks=[];handles=[]
    prefixes=tuple(f'runs/{kind}/{arm}' for kind in ('source_extensions','source_extension_downstream') for arm in ARMS)
    for prefix in prefixes:
        base=checked_path(root,prefix)
        if not base.exists():continue
        for path in sorted(base.rglob('*')):
            checked_path(root,path.relative_to(root).as_posix())
            if path.is_file():
                if path.suffix=='.lock':locks.append(path)
                elif path.suffix in {'.json','.npy','.pt'} and path.name!='backup_manifest.json':candidates.append(path)
    claims=checked_path(root,'runs/source_extensions/claims')
    if claims.exists():
        for arm in ARMS:
            for path in sorted(claims.glob(arm+'-seed-*')):
                checked_path(root,path.relative_to(root).as_posix())
                if path.is_file():
                    if path.suffix=='.lock':locks.append(path)
                    elif path.suffix=='.json':candidates.append(path)
    if not candidates:raise BootstrapError('no_actual_owned_extension_outputs_to_snapshot')
    try:
        for path in sorted(set(locks)):
            handle=path.open('rb');handles.append(handle)
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise BootstrapError('stop_owned_extension_workers_before_result_snapshot') from None
        files=[{**file_ref(root,path),'role':'result','dataset':'Paper3SourceExtension'} for path in sorted(set(candidates))]
        value={'schema_version':1,'files':files,'operation':'paper3_owned_extension_result_snapshot',
            'upload_performed':False,'full_GET_verified':False,'scientific_completion_claimed':False,
            'contains_credentials':False,'fixed_result_namespaces':list(prefixes)}
        target=f'audit/deployment_result_snapshots/{digest(value)}_source_extensions.json'
        result={'status':'LOCAL_RESULT_SNAPSHOT_PLANNED_NO_NETWORK','manifest_path':target,
            'file_count':len(files),'bytes':sum(r['bytes'] for r in files),'upload_performed':False,
            'scientific_completion_claimed':False,'fit_claims_deleted':False}
        if execute:
            write_once(checked_path(root,target),value)
            result.update(status='LOCAL_IMMUTABLE_RESULT_SNAPSHOT_WRITTEN_NOT_UPLOADED',manifest=file_ref(root,target))
        return result
    finally:
        for handle in handles:handle.close()


def prepare(root,index_path,*,execute=False,launch=False,user_started_gpu=False,runner=subprocess.run):
    root=Path(root).resolve(strict=True);index_path=Path(index_path)
    index_path=index_path if index_path.is_absolute() else checked_path(root,str(index_path),exists=True)
    if not index_path.is_relative_to(root):raise BootstrapError('bootstrap_index_must_be_inside_fixed_package')
    index_path=checked_path(root,index_path.relative_to(root).as_posix(),exists=True)
    index=validate_index(root,index_path)
    if launch and (not execute or not user_started_gpu):raise BootstrapError('launch_requires_execute_and_later_user_started_GPU_flags')
    result={'schema_version':SCHEMA,'status':'DRY_RUN_FIXED_INDEX_AND_EMBEDDED_FILES_VERIFIED',
        'index_sha256':index['index_sha256'],'file_count':len(index['files']),
        'raw_payloads_selected':0,'network_operations':0,'scientific_fits':0,
        'GPU_checked':False,'server_actions':False,'embedded_files':embedded_files(index,index_path.relative_to(root).as_posix())}
    if not execute:return result
    parent=checked_path(root,STATE);parent.mkdir(exist_ok=True)
    owner=checked_path(root,STATE+'/owner.lock').open('a')
    try:
        try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise BootstrapError('one_local_bootstrap_controller_already_active') from None
        disk_preflight(root,index)
        python=install_sdk(root,index,runner=runner);env=thread_environment()
        run_command([python,'tools/bootstrap_data_plan.py','--index',index_path.relative_to(root),'--restore-worker','--execute'],
            root=root,env=env,log=root/STATE/'restore_worker.log',runner=runner)
        install_full_profile(root,index,python,runner=runner)
        result.update(status='CACHE_AND_OFFLINE_ENVIRONMENT_READY_GPU_NOT_CHECKED',network_operations='see_per_group_full_GET_receipts',
            environment_python=str(python))
        if launch:result=launch_training(root,index,python,user_started_gpu=user_started_gpu,runner=runner)
        return result
    finally:owner.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path)
    parser.add_argument('--snapshot-extension-results',action='store_true',
                        help='Finite local result manifest only; --execute writes it, no R2 or fit')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--launch-training',action='store_true')
    parser.add_argument('--user-started-gpu',action='store_true')
    parser.add_argument('--restore-worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    try:
        if args.snapshot_extension_results:
            if args.index or args.restore_worker or args.launch_training or args.user_started_gpu:
                raise BootstrapError('result_snapshot_must_be_separate_from_restore_or_training')
            result=snapshot_extension_results(ROOT,execute=args.execute)
        elif args.index is None:raise BootstrapError('fixed_bootstrap_index_required')
        elif args.restore_worker:
            if not args.execute or args.launch_training:raise BootstrapError('restore_worker_execution_scope_invalid')
            result=restore_files(ROOT,validate_index(ROOT,ROOT/args.index))
        else:result=prepare(ROOT,args.index,execute=args.execute,launch=args.launch_training,user_started_gpu=args.user_started_gpu)
        print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False));return 0
    except Exception as exc:
        print(json.dumps({'status':'BLOCKED_PRESERVE_EXISTING_DATA_AND_JOBS','error_type':type(exc).__name__,
            'reason':str(exc) if isinstance(exc,BootstrapError) else 'local_bootstrap_failed_details_suppressed',
            'server_actions':False,'scientific_completion_claimed':False},indent=2));return 2


if __name__=='__main__':raise SystemExit(main())
