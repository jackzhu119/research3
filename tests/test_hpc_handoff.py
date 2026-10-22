"""Finite offline snapshots and original full-GET proofs; no real network/fit."""
import fcntl
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hpc import handoff
from tools import r2_backup, r2_restore


class Bootstrap:
    class BootstrapError(ValueError):
        pass

    def snapshot_extension_results(self, root, execute):
        assert execute is True
        raise self.BootstrapError('no_actual_owned_extension_outputs_to_snapshot')


class Campaign:
    def __init__(self, rows): self.rows = rows
    def collect_backup_manifest(self, root):
        return {'schema_version': 1, 'files': self.rows}


class Conflict(Exception):
    response = {'ResponseMetadata': {'HTTPStatusCode': 412}}


class Body(io.BytesIO):
    def __init__(self, value): super().__init__(value); self.eof = False
    def read(self, size=-1):
        value = super().read(size)
        if value == b'': self.eof = True
        return value


class S3:
    def __init__(self): self.objects = {}; self.puts = []; self.gets = []; self.bodies = []; self.corrupt = False
    def put_object(self, **kwargs):
        assert kwargs['IfNoneMatch'] == '*'
        self.puts.append(kwargs['Key'])
        if kwargs['Key'] in self.objects: raise Conflict('do not print sensitive SDK detail')
        raw = kwargs['Body'].read(); assert len(raw) == kwargs['ContentLength']
        self.objects[kwargs['Key']] = raw
        return {'ResponseMetadata': {'HTTPStatusCode': 200}}
    def get_object(self, **kwargs):
        self.gets.append(kwargs['Key']); raw = self.objects[kwargs['Key']]
        if self.corrupt: raw = bytes([raw[0] ^ 1]) + raw[1:]
        body = Body(raw); self.bodies.append(body)
        return {'Body': body, 'ContentLength': len(raw), 'ResponseMetadata': {'HTTPStatusCode': 200}}


def inputs(tmp_path, *, path='runs/source_training_v2/seed-1/model.pt', data=b'fixture-checkpoint-actual-bytes'):
    source = tmp_path / path; source.parent.mkdir(parents=True, exist_ok=True); source.write_bytes(data)
    row = {'path': path, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
           'role': 'result', 'dataset': 'Paper3_mainline'}
    apis = (Bootstrap(), Campaign([row]), r2_backup, r2_restore)
    return source, row, apis


def queue_path(root, result): return root / result['queue_path']


def pilot_runtime_fixture(root, *, school=True, failed=False, with_identity=True):
    """Metadata-only SOFTWARE_TEST_DOUBLE, never a GPU completion proof."""
    from hpc import runtime_profile
    from p3work import training_campaign
    folder = root / handoff.PILOT; folder.mkdir(parents=True)
    audit = root / 'audit'; audit.mkdir()
    pins = dict(runtime_profile.REGISTRY['pins']) if school else {'torch':'2.8.0+cu128','numpy':'2.3.5'}
    lock = root / 'requirements-gpu.lock'
    lock.write_text('torch==2.8.0+cu128\nnumpy==2.3.5\n')
    handoff.write_json(root / handoff.PILOT_RUNTIME_AUDITS[0],
                       [{'name': name, 'version': version} for name,version in sorted(pins.items())])
    def descriptor(relative, *, size=True):
        sha,n=handoff.hash_file(root/relative)
        return {'path':relative,'sha256':sha,**({'bytes':n} if size else {})}
    dependencies=descriptor(handoff.PILOT_RUNTIME_AUDITS[0])
    profile=None
    if school:
        proof={'schema_version':'paper3-hpc-installed-runtime-profile-v1',
            'profile':runtime_profile.PROFILE,'project':str(root),'python':'3.12.3',
            'python_executable':'SOFTWARE_TEST_DOUBLE_NOT_AN_EXECUTABLE',
            'registered_profile_sha256':hashlib.sha256(handoff.json_bytes(runtime_profile.REGISTRY)).hexdigest(),
            'original_RunPod_lock_sha256':hashlib.sha256(lock.read_bytes()).hexdigest(),
            'resolved_dependencies':descriptor(handoff.PILOT_RUNTIME_AUDITS[0],size=False),
            'inventory_packages':len(pins),'actual_module_versions':dict(runtime_profile.REGISTRY['pins']),
            'packages_installed_by_capture':0,'network_operations_by_capture':0,
            'scientific_parameters_changed':False}
        handoff.write_json(root/handoff.PILOT_RUNTIME_AUDITS[1],proof)
        profile=descriptor(handoff.PILOT_RUNTIME_AUDITS[1],size=False)
    claim=root/'runs/controlled_pilot_job_claim.json'
    handoff.write_json(claim,{'scope':'ENGINEERING_SOURCE_PILOT_ONLY','max_jobs':1,'output':handoff.PILOT})
    identity={'resolved_dependencies':dependencies,'torch':pins['torch'],'numpy':pins['numpy'],
              'fixture_scope':'SOFTWARE_TEST_DOUBLE_NO_FIT'}
    receipt={'scope':'ENGINEERING_SOURCE_PILOT_ONLY',
        'status':'ENGINEERING_PILOT_FAILED' if failed else 'ENGINEERING_PILOT_COMPLETE',
        'training_started':not failed,'actual_completed_updates':0 if failed else 200,
        'optimizer_updates_dispatched':0 if failed else 200,'resolved_dependencies':dependencies,
        'fixture_scope':'SOFTWARE_TEST_DOUBLE_METADATA_NOT_A_GPU_PROOF'}
    if profile is not None:
        identity['deployment_runtime_profile']=profile;receipt['deployment_runtime_profile']=profile
    if with_identity:handoff.write_json(folder/'identity.json',identity)
    if not failed:
        (folder/'engineering_initial.pt').write_bytes(b'SOFTWARE_TEST_DOUBLE_CHECKPOINT_NO_MODEL')
    def seal():
        paths=[claim,root/handoff.PILOT_RUNTIME_AUDITS[0]]
        if school:paths.append(root/handoff.PILOT_RUNTIME_AUDITS[1])
        paths.extend(path for path in folder.iterdir() if path.name in {'identity.json','engineering_initial.pt'})
        rows=[{**descriptor(path.relative_to(root).as_posix()),'role':'result','dataset':'Paper3SourceEngineering'}
              for path in paths]
        artifact=handoff.PILOT+'/engineering_artifact_manifest.json'
        handoff.write_json(root/artifact,{'schema_version':'paper3-source-engineering-artifacts-v1',
            'scope':'ENGINEERING_SOURCE_PILOT_ONLY','scientific_result':False,
            'pilot_status':receipt['status'],'files':rows})
        receipt['artifact_manifest']=descriptor(artifact)
        handoff.write_json(folder/'engineering_receipt.json',receipt)
    seal()
    return (Bootstrap(),training_campaign,r2_backup,r2_restore), receipt, descriptor, seal


@pytest.mark.parametrize('school',[False,True])
@pytest.mark.parametrize('failed',[False,True])
def test_bound_pilot_dependency_provenance_survives_offline_handoff_and_full_get(tmp_path,school,failed):
    apis,receipt,descriptor,seal=pilot_runtime_fixture(tmp_path,school=school,failed=failed)
    before={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    snap=handoff.snapshot(tmp_path,job_exit_code=2 if failed else 0,apis=apis)
    payload=queue_path(tmp_path,snap)/'payload'
    expected=handoff.PILOT_RUNTIME_AUDITS if school else handoff.PILOT_RUNTIME_AUDITS[:1]
    for name in expected:
        assert (payload/name).read_bytes()==before[name]
    assert snap['status']=='PENDING_EXTERNAL_BACKUP_NOT_UPLOADED' and not snap['uploaded']
    assert all((tmp_path/name).read_bytes()==raw for name,raw in before.items())
    client=S3(); result=handoff.flush(tmp_path,client=client,bucket='SOFTWARE_TEST_DOUBLE',apis=apis)
    assert result['uploaded'] and all(body.eof and body.closed for body in client.bodies)
    for name in expected:
        assert client.objects['paper3/hpc-training-results/'+snap['queue_id']+'/'+name]==before[name]


def test_prefit_failed_pilot_without_identity_can_retain_actual_bound_runtime_metadata(tmp_path):
    apis,_,_,_=pilot_runtime_fixture(tmp_path,failed=True,with_identity=False)
    snap=handoff.snapshot(tmp_path,job_exit_code=2,apis=apis)
    assert all((queue_path(tmp_path,snap)/'payload'/name).is_file() for name in handoff.PILOT_RUNTIME_AUDITS)
    assert not (tmp_path/handoff.PILOT/'identity.json').exists()


@pytest.mark.parametrize('name',handoff.PILOT_RUNTIME_AUDITS)
def test_changed_pilot_runtime_proof_blocks_snapshot_without_overwriting_originals(tmp_path,name):
    apis,_,_,_=pilot_runtime_fixture(tmp_path)
    path=tmp_path/name;path.write_bytes(path.read_bytes()+b'\n');raw=path.read_bytes()
    with pytest.raises(handoff.HandoffError,match='artifact_SHA_mismatch'):
        handoff.snapshot(tmp_path,apis=apis)
    assert path.read_bytes()==raw and not (tmp_path/handoff.BASE).exists()


def test_self_consistent_forged_hpc_runtime_registry_is_not_backup_authority(tmp_path):
    apis,receipt,descriptor,seal=pilot_runtime_fixture(tmp_path)
    path=tmp_path/handoff.PILOT_RUNTIME_AUDITS[1]
    proof=json.loads(path.read_bytes());proof['actual_module_versions']['torch']='2.8.0+cu128'
    handoff.write_json(path,proof)
    receipt['deployment_runtime_profile']=descriptor(handoff.PILOT_RUNTIME_AUDITS[1],size=False)
    identity_path=tmp_path/handoff.PILOT/'identity.json';identity=json.loads(identity_path.read_bytes())
    identity['deployment_runtime_profile']=receipt['deployment_runtime_profile'];handoff.write_json(identity_path,identity);seal()
    with pytest.raises(handoff.HandoffError,match='registered_provenance_mismatch'):
        handoff.snapshot(tmp_path,apis=apis)


@pytest.mark.parametrize('mode',['reference','manifest'])
def test_pilot_runtime_references_cannot_inject_env_or_other_audit_files(tmp_path,monkeypatch,mode):
    apis,receipt,descriptor,seal=pilot_runtime_fixture(tmp_path)
    secret=tmp_path/'.env';secret.write_bytes(b'SOFTWARE_TEST_DOUBLE_SECRET_MUST_NOT_BE_READ')
    if mode=='reference':
        receipt['deployment_runtime_profile']=descriptor('.env',size=False)
        identity_path=tmp_path/handoff.PILOT/'identity.json';identity=json.loads(identity_path.read_bytes())
        identity['deployment_runtime_profile']=receipt['deployment_runtime_profile'];handoff.write_json(identity_path,identity);seal()
    else:
        artifact=tmp_path/handoff.PILOT/'engineering_artifact_manifest.json'
        value=json.loads(artifact.read_bytes());value['files'].append({**descriptor('.env'),
            'role':'result','dataset':'Paper3SourceEngineering'})
        handoff.write_json(artifact,value)
        receipt['artifact_manifest']=descriptor(artifact.relative_to(tmp_path).as_posix())
        handoff.write_json(tmp_path/handoff.PILOT/'engineering_receipt.json',receipt)
    actual=handoff.hash_file
    def safe_hash(path):
        assert Path(path)!=secret,'runtime closure tried to read the .env file'
        return actual(path)
    monkeypatch.setattr(handoff,'hash_file',safe_hash)
    with pytest.raises(handoff.HandoffError):handoff.snapshot(tmp_path,apis=apis)
    assert not (tmp_path/handoff.BASE).exists()


def test_offline_snapshot_copies_real_bytes_and_never_uploads(tmp_path, monkeypatch):
    source, row, apis = inputs(tmp_path)
    monkeypatch.setattr(r2_backup, 'run_backup', lambda **kwargs: pytest.fail('snapshot must not call network API'))
    result = handoff.snapshot(tmp_path, label='train', job_exit_code=2, apis=apis)
    assert result['status'] == 'PENDING_EXTERNAL_BACKUP_NOT_UPLOADED' and result['uploaded'] is False
    assert result['training_dispatched'] is False and result['job_exit_code'] == 2
    assert (queue_path(tmp_path, result) / 'payload' / row['path']).read_bytes() == source.read_bytes()
    assert source.exists()


def test_snapshot_deduplicates_content_and_retains_prior_events(tmp_path):
    source, row, apis = inputs(tmp_path)
    first = handoff.snapshot(tmp_path, apis=apis)
    second = handoff.snapshot(tmp_path, label='preempted', apis=apis)
    assert first['queue_id'] == second['queue_id']
    assert len(list((queue_path(tmp_path, first) / 'snapshot_events').glob('*.json'))) == 2
    assert source.read_bytes() == b'fixture-checkpoint-actual-bytes'


def test_empty_actual_producer_has_no_queue_or_upload_claim(tmp_path):
    apis = (Bootstrap(), Campaign([]), r2_backup, r2_restore)
    result = handoff.snapshot(tmp_path, apis=apis)
    assert result['status'] == 'NO_ACTUAL_OWNED_RESULTS_TO_SNAPSHOT' and not result['uploaded']
    result = handoff.flush(tmp_path, apis=apis, prompt=True)
    assert result['status'] == 'NO_PENDING_HANDOFFS' and not result['uploaded']


@pytest.mark.parametrize('path', ['cache/model.pt', 'runs/source_training_v2/data.npz',
                                 'runs/source_training_v2/../../paper1/model.pt', 'runs/sealed/query.npy'])
def test_non_original_result_scope_is_rejected(tmp_path, path):
    if '..' in path:
        source, row, apis = inputs(tmp_path)
        apis[1].rows[0]['path'] = path
    else:
        source, row, apis = inputs(tmp_path, path=path)
    with pytest.raises(handoff.HandoffError): handoff.snapshot(tmp_path, apis=apis)
    assert source.exists()


@pytest.mark.parametrize('lock', ['.paper3-deployment/owner.lock', 'runs/.training_campaign_v1.lock'])
def test_running_original_controller_blocks_snapshot(tmp_path, lock):
    source, row, apis = inputs(tmp_path)
    path = tmp_path / lock; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(handoff.HandoffError, match='active_controller'):
            handoff.snapshot(tmp_path, apis=apis)
    assert source.exists()


def test_duplicate_results_are_not_silently_merged(tmp_path):
    source, row, apis = inputs(tmp_path); apis[1].rows.append(dict(row))
    with pytest.raises(handoff.HandoffError, match='duplicate_owned_result_path'):
        handoff.snapshot(tmp_path, apis=apis)


def test_queued_bytes_survive_newer_source_checkpoint_and_full_get_is_required(tmp_path):
    source, row, apis = inputs(tmp_path); original = source.read_bytes()
    snap = handoff.snapshot(tmp_path, apis=apis); source.write_bytes(b'newer-live-checkpoint')
    client = S3(); result = handoff.flush(tmp_path, client=client, bucket='fixture-bucket', apis=apis)
    assert result['uploaded'] and result['status'] == 'ALL_SELECTED_HANDOFFS_FULL_GET_VERIFIED'
    assert client.objects['paper3/hpc-training-results/' + snap['queue_id'] + '/' + row['path']] == original
    assert source.read_bytes() == b'newer-live-checkpoint'
    assert all(body.eof and body.closed for body in client.bodies)
    receipt = json.loads((tmp_path / result['queues'][0]['receipt_path']).read_bytes())
    rows, digest = r2_restore.load_verified_receipt(tmp_path / result['queues'][0]['receipt_path'])
    assert any(item['path'] == row['path'] and item['sha256'] == row['sha256'] for item in rows)
    assert len(rows) == 2  # Checkpoint and this real collection's immutable manifest.
    assert receipt['verification'] == 'full_get_body_sha256'


def test_failed_full_get_keeps_pending_then_retry_only_backups(tmp_path):
    source, row, apis = inputs(tmp_path); snap = handoff.snapshot(tmp_path, apis=apis)
    client = S3(); client.corrupt = True
    first = handoff.flush(tmp_path, client=client, bucket='fixture', apis=apis)
    assert first['uploaded'] is False and first['status'] == 'HANDOFFS_PENDING_RETRY_BACKUP_ONLY'
    assert source.exists() and (queue_path(tmp_path, snap) / 'payload' / row['path']).exists()
    client.corrupt = False
    second = handoff.flush(tmp_path, client=client, bucket='fixture', apis=apis)
    assert second['uploaded'] is True and len(client.gets) == 3
    assert len(list((queue_path(tmp_path, snap) / 'attempts').iterdir())) == 2
    assert second['training_dispatched'] is False


def test_successful_reflush_repeats_actual_complete_get(tmp_path):
    source, row, apis = inputs(tmp_path); handoff.snapshot(tmp_path, apis=apis); client = S3()
    assert handoff.flush(tmp_path, client=client, bucket='fixture', apis=apis)['uploaded']
    assert handoff.flush(tmp_path, client=client, bucket='fixture', apis=apis)['uploaded']
    assert len(client.gets) == 4 and all(body.eof for body in client.bodies)


def test_corrupt_queue_is_rejected_before_network(tmp_path):
    source, row, apis = inputs(tmp_path); snap = handoff.snapshot(tmp_path, apis=apis)
    (queue_path(tmp_path, snap) / 'payload' / row['path']).write_bytes(b'changed')
    client = S3()
    with pytest.raises(handoff.HandoffError, match='snapshot_sha_mismatch'):
        handoff.flush(tmp_path, client=client, bucket='fixture', apis=apis)
    assert not client.puts and not client.gets and source.exists()


def test_payload_root_symlink_is_rejected(tmp_path):
    source, row, apis = inputs(tmp_path); snap = handoff.snapshot(tmp_path, apis=apis)
    payload = queue_path(tmp_path, snap) / 'payload'; moved = tmp_path / 'elsewhere'; payload.rename(moved)
    payload.symlink_to(moved, target_is_directory=True)
    with pytest.raises(handoff.HandoffError, match='symlink_forbidden'):
        handoff.flush(tmp_path, client=S3(), bucket='fixture', apis=apis)


def test_original_extension_snapshot_and_copy_keep_worker_lock_without_self_deadlock(tmp_path, monkeypatch):
    from tools import bootstrap_data_plan
    source, row, apis = inputs(tmp_path)
    ext = tmp_path / 'runs/source_extensions/BASE4_PLUS_YANG/seed-1'
    ext.mkdir(parents=True); (ext / 'checkpoint.pt').write_bytes(b'fixture-extension')
    worker_lock = ext / '.worker.lock'; worker_lock.write_text('')
    apis = (bootstrap_data_plan, apis[1], r2_backup, r2_restore)
    original = handoff.shutil.copyfileobj
    calls = []
    def guarded_copy(src, dst, size):
        with worker_lock.open('a') as handle:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        calls.append(True); return original(src, dst, size)
    monkeypatch.setattr(handoff.shutil, 'copyfileobj', guarded_copy)
    snap = handoff.snapshot(tmp_path, apis=apis)
    assert snap['file_count'] == 4 and len(calls) == 4
    assert (queue_path(tmp_path, snap) / 'payload' / ext.relative_to(tmp_path) / 'checkpoint.pt').read_bytes() == b'fixture-extension'


def test_original_busy_extension_worker_blocks_handoff(tmp_path):
    from tools import bootstrap_data_plan
    source, row, apis = inputs(tmp_path)
    ext = tmp_path / 'runs/source_extensions/BASE4_PLUS_DREYER/seed-1'
    ext.mkdir(parents=True); (ext / 'checkpoint.pt').write_bytes(b'fixture-extension')
    with (ext / '.worker.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(bootstrap_data_plan.BootstrapError, match='stop_owned_extension_workers'):
            handoff.snapshot(tmp_path, apis=(bootstrap_data_plan, apis[1], r2_backup, r2_restore))
    assert source.exists()


def test_live_campaign_resume_metadata_and_new_partial_checkpoint_restore_exactly(tmp_path):
    from p3work import training_campaign
    root = tmp_path / 'original'; root.mkdir()
    model = root / 'runs/source_training_v2/seed-1/fit/latest.pt'
    model.parent.mkdir(parents=True); model.write_bytes(b'fixture-old-checkpoint')
    claim = root / 'runs/training_campaign_v1_claim.json'
    claim.write_bytes(b'{"fixture_only":true,"identity":"unchanged-claim"}\n')
    wall = root / 'runs/training_campaign_v1/wall_budget.json'
    wall.parent.mkdir(parents=True); wall.write_bytes(b'{"fixture_only":true,"charged_seconds":71}\n')
    original_ref = training_campaign.backup_manifest(root)
    original_manifest = root / original_ref['path']
    old_bytes = original_manifest.read_bytes()
    progress = root / 'runs/training_campaign_v1/campaign_progress.json'
    progress.write_text(json.dumps({'fixture_only': True, 'backup_manifest': original_ref, 'actions': []}))
    # A newer partial checkpoint must be backed up without rewriting the older
    # producer document to which this actual live state still points.
    model.write_bytes(b'fixture-newer-partial-checkpoint')
    expected = {p.relative_to(root).as_posix(): p.read_bytes() for p in (model, claim, wall, progress, original_manifest)}
    apis = (Bootstrap(), training_campaign, r2_backup, r2_restore)
    snapshot = handoff.snapshot(root, label='interrupted', job_exit_code=3, apis=apis)
    queue = queue_path(root, snapshot)
    manifest = json.loads((queue / 'manifest.json').read_bytes())
    payload = queue / 'payload'
    for path, raw in expected.items():
        assert (payload / path).read_bytes() == raw == (root / path).read_bytes()
    assert original_manifest.read_bytes() == old_bytes
    collections = [row for row in manifest['files'] if row['path'].startswith('.paper3-hpc/producer_manifests/')]
    assert len(collections) == 1 and Path(collections[0]['path']).stem == collections[0]['sha256']
    collected = json.loads((payload / collections[0]['path']).read_bytes())
    checkpoint_row = next(row for row in collected['files'] if row['path'] == model.relative_to(root).as_posix())
    assert checkpoint_row['sha256'] == hashlib.sha256(expected[model.relative_to(root).as_posix()]).hexdigest()
    assert all(row['path'] != progress.relative_to(root).as_posix() for row in collected['files'])
    client = S3()
    flushed = handoff.flush(root, client=client, bucket='fixture', apis=apis)
    assert flushed['uploaded'] and flushed['objects_verified'] == len(manifest['files'])
    receipt_path = root / flushed['queues'][0]['receipt_path']
    restored = tmp_path / 'restored'
    result = r2_restore.run_restore(backup_receipt_path=receipt_path, root=restored,
        report_path=tmp_path / 'restore_report.json', all_approved=True, execute=True, client=client, bucket='fixture')
    assert result['status'] == 'restored_verified'
    for path, raw in expected.items():
        assert (restored / path).read_bytes() == raw
    state = json.loads((restored / progress.relative_to(root)).read_bytes())
    restored_old_manifest = restored / state['backup_manifest']['path']
    assert hashlib.sha256(restored_old_manifest.read_bytes()).hexdigest() == state['backup_manifest']['sha256']
    assert all(body.eof and body.closed for body in client.bodies)


@pytest.mark.parametrize('existing', [False, True])
def test_missing_or_changed_manifest_referenced_by_live_progress_is_not_fabricated(tmp_path, existing):
    from p3work import training_campaign
    model = tmp_path / 'runs/source_training_v2/seed-1/model.pt'
    model.parent.mkdir(parents=True); model.write_bytes(b'fixture-partial-checkpoint')
    progress = tmp_path / 'runs/training_campaign_v1/campaign_progress.json'
    progress.parent.mkdir(parents=True)
    reference = {'path': 'runs/training_campaign_v1/backup_manifest.json', 'sha256': '0' * 64}
    progress.write_text(json.dumps({'backup_manifest': reference, 'fixture_only': True}))
    old = tmp_path / reference['path']
    if existing: old.write_bytes(b'{"schema_version":1,"files":[]}\n')
    before = progress.read_bytes(), model.read_bytes(), old.read_bytes() if existing else None
    apis = (Bootstrap(), training_campaign, r2_backup, r2_restore)
    with pytest.raises(handoff.HandoffError, match='live_progress_original_manifest_'):
        handoff.snapshot(tmp_path, apis=apis)
    assert (progress.read_bytes(), model.read_bytes(), old.read_bytes() if existing else None) == before
    assert old.exists() is existing


def test_original_collection_is_pure_and_runpod_writer_retains_original_value(tmp_path):
    from p3work import training_campaign
    source, row, _ = inputs(tmp_path)
    expected = training_campaign.collect_backup_manifest(tmp_path)
    original_path = tmp_path / 'runs/training_campaign_v1/backup_manifest.json'
    assert not original_path.exists()
    reference = training_campaign.backup_manifest(tmp_path)
    raw = original_path.read_bytes()
    assert json.loads(raw) == expected and reference['sha256'] == hashlib.sha256(raw).hexdigest()
    assert training_campaign.collect_backup_manifest(tmp_path) == expected and original_path.read_bytes() == raw


def test_extension_snapshot_keeps_original_manifest_documents_and_full_get_binding(tmp_path):
    from p3work import training_campaign
    from tools import bootstrap_data_plan
    extension = tmp_path / 'runs/source_extensions/BASE4_PLUS_YANG/seed-1'
    extension.mkdir(parents=True)
    (extension / 'checkpoint.pt').write_bytes(b'fixture-extension-weight')
    manifest = extension / 'backup_manifest.json'
    manifest.write_bytes(b'{"schema_version":1,"files":[],"fixture_only":true}\n')
    progress = extension / 'progress.json'
    progress.write_text(json.dumps({'fixture_only':True, 'backup_manifest':{
        'path':manifest.relative_to(tmp_path).as_posix(), 'sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}}))
    apis = (bootstrap_data_plan, training_campaign, r2_backup, r2_restore)
    result = handoff.snapshot(tmp_path, apis=apis)
    queue = queue_path(tmp_path, result)
    selected = json.loads((queue / 'manifest.json').read_bytes())['files']
    paths = {row['path'] for row in selected}
    assert manifest.relative_to(tmp_path).as_posix() in paths
    assert progress.relative_to(tmp_path).as_posix() in paths
    source_refs = [p for p in paths if p.startswith('audit/deployment_result_snapshots/')]
    assert len(source_refs) == 1
    for path in paths:
        assert (queue / 'payload' / path).read_bytes() == (tmp_path / path).read_bytes()
    client = S3(); flushed = handoff.flush(tmp_path, client=client, bucket='fixture', apis=apis)
    assert flushed['uploaded'] and flushed['objects_verified'] == len(paths)
    rows, _ = r2_restore.load_verified_receipt(tmp_path / flushed['queues'][0]['receipt_path'])
    assert {tuple(row[k] for k in handoff.FIELDS) for row in rows} == {
        tuple(row[k] for k in handoff.FIELDS) for row in selected}


TOKEN = 'fixture-pat-must-not-persist'
HEAD = 'a' * 40


class Git:
    def __init__(self, *, remote=None, paths=None, dirty=False, blobs=None):
        self.remote = remote; self.paths = paths or ['README.md', 'p3work/module.py', 'hpc/handoff.py']
        self.dirty = dirty; self.calls = []; self.helpers = []; self.pushes = 0; self.blobs = blobs or {}
    def __call__(self, argv, **kwargs):
        assert TOKEN not in repr(argv); assert '-A' not in argv and '--force' not in argv
        assert kwargs['timeout'] == 300
        args = argv[7:]; self.calls.append(args)
        assert args[0] not in {'add', 'commit', 'checkout', 'reset'}
        if 'GIT_ASKPASS' in kwargs['env']:
            path = Path(kwargs['env']['GIT_ASKPASS']); self.helpers.append(path)
            assert TOKEN not in path.read_text()
            assert kwargs['env']['PAPER3_HPC_GITHUB_TOKEN'] == TOKEN
        assert kwargs['env'].get('HTTPS_PROXY') == 'https://proxy.invalid'
        if args == ['rev-parse', 'HEAD']: out = HEAD
        elif args[:1] == ['rev-parse'] and args[1].startswith('HEAD:'): out = self.blobs.get(args[1][5:], '0' * 40)
        elif args[:1] == ['diff']: out = 'p3work/module.py' if self.dirty else ''
        elif args[:1] == ['ls-tree']: out = '\n'.join(self.paths)
        elif args[:1] == ['ls-remote']: out = '' if self.remote is None else self.remote + '\t' + args[-1]
        elif args[:1] == ['push']:
            self.pushes += 1; self.remote = HEAD; out = ''
        else: pytest.fail('unexpected Git command ' + repr(args))
        return subprocess.CompletedProcess(argv, 0, out, '')


@pytest.fixture
def git_env(monkeypatch):
    monkeypatch.setattr(handoff.os, 'environ', {'PATH':'/usr/bin:/bin', 'HTTPS_PROXY':'https://proxy.invalid',
         'AWS_SECRET_ACCESS_KEY':'fixture-r2-secret', 'GIT_TRACE':'fixture-bad-trace', 'GITHUB_TOKEN':'fixture-old-token'})


def test_github_pushes_only_committed_code_and_verifies_remote_sha(tmp_path, git_env, capsys):
    git = Git(); value = handoff.publish_code(tmp_path, token=TOKEN, runner=git)
    assert value['actual_push_performed'] and git.pushes == 1 and value['commit'] == HEAD
    assert all(not path.exists() for path in git.helpers)
    assert TOKEN not in json.dumps(value) and TOKEN not in capsys.readouterr().out


def test_github_existing_equal_remote_does_not_push_again(tmp_path, git_env):
    git = Git(remote=HEAD); value = handoff.publish_code(tmp_path, token=TOKEN, runner=git)
    assert not value['actual_push_performed'] and git.pushes == 0


@pytest.mark.parametrize('path', ['data/raw.csv', 'cache/source.npz', 'audit/private.json',
    'protocol/Paper3.docx', 'runs/model.pt', '.env', 'unexpected-secret.txt'])
def test_github_committed_data_or_unapproved_file_is_rejected(tmp_path, git_env, path):
    git = Git(paths=[path])
    with pytest.raises(handoff.HandoffError, match='github_data_or_credential_file_forbidden'):
        handoff.publish_code(tmp_path, token=TOKEN, runner=git)
    assert not git.pushes


def test_github_different_remote_is_preserved(tmp_path, git_env):
    git = Git(remote='b' * 40)
    with pytest.raises(handoff.HandoffError, match='remote_branch_differs_preserved'):
        handoff.publish_code(tmp_path, token=TOKEN, runner=git)
    assert git.pushes == 0


def test_github_tracked_changes_require_manual_reviewed_commit(tmp_path, git_env):
    git = Git(dirty=True)
    with pytest.raises(handoff.HandoffError, match='tracked_changes_require_manual_commit'):
        handoff.publish_code(tmp_path, token=TOKEN, runner=git)
    assert git.pushes == 0


def test_github_failure_does_not_erase_actual_r2_success(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(handoff, 'flush', lambda *a, **k: {'status':'ALL_SELECTED_HANDOFFS_FULL_GET_VERIFIED', 'uploaded':True})
    def failed(*a, **k): raise handoff.HandoffError('github_permission_not_ready')
    monkeypatch.setattr(handoff, 'publish_code', failed)
    assert handoff.main(['--root', str(tmp_path), '--flush', '--publish-code']) == 2
    value = json.loads(capsys.readouterr().out)
    assert value['uploaded'] is True and value['github']['status'] == 'GITHUB_PUBLICATION_PENDING'


def test_github_network_timeout_is_bounded_and_does_not_emit_sdk_detail(tmp_path, git_env):
    def blocked(argv, **kwargs):
        assert kwargs['timeout'] == 300
        raise subprocess.TimeoutExpired(argv, kwargs['timeout'], output=TOKEN, stderr='fixture-private-detail')
    with pytest.raises(handoff.HandoffError, match='^github_git_operation_failed$'):
        handoff.publish_code(tmp_path, token=TOKEN, runner=blocked)


def test_hidden_input_refuses_non_tty(monkeypatch):
    monkeypatch.setattr(handoff.sys, 'stdin', SimpleNamespace(isatty=lambda:False))
    with pytest.raises(handoff.HandoffError, match='interactive_hidden_input_required'):
        handoff.hidden('fixture')


@pytest.mark.parametrize('path,blob', [
    ('hpc/fixtures/original_source_training_54b972.py.txt', 'fee812b6165101afc6b0141ce5e0a8f0d158df85'),
    ('hpc/fixtures/original_run_controlled_pilot_856393.py.txt', '793c40fe37a9c0d7f8826232a6e7622e183377c2'),
])
def test_github_historical_code_snapshot_requires_exact_original_committed_blob(tmp_path, git_env, path, blob):
    git = Git(paths=[path], blobs={path:blob})
    assert handoff.publish_code(tmp_path, token=TOKEN, runner=git)['actual_push_performed']
    tampered = Git(paths=[path], blobs={path:'b' * 40})
    with pytest.raises(handoff.HandoffError, match='github_data_or_credential_file_forbidden'):
        handoff.publish_code(tmp_path, token=TOKEN, runner=tampered)
    assert tampered.pushes == 0


def test_github_does_not_admit_other_historical_text_files(tmp_path, git_env):
    git = Git(paths=['hpc/fixtures/another_code_snapshot.py.txt'])
    with pytest.raises(handoff.HandoffError, match='github_data_or_credential_file_forbidden'):
        handoff.publish_code(tmp_path, token=TOKEN, runner=git)
    assert git.pushes == 0
