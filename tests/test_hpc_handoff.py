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
    def __init__(self, *, remote=None, paths=None, dirty=False):
        self.remote = remote; self.paths = paths or ['README.md', 'p3work/module.py', 'hpc/handoff.py']
        self.dirty = dirty; self.calls = []; self.helpers = []; self.pushes = 0
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
