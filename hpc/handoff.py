#!/usr/bin/env python3
"""Durable offline result handoff; upload only when explicitly requested.

Compute jobs snapshot the original producers' finite result namespaces. Login
or transfer jobs may later flush those immutable bytes with the original R2
PUT/full-GET implementation. This module never dispatches a scientific fit.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
import warnings

MAIN = ('runs/training_campaign_v1', 'runs/controlled_source_pilot_seed20261018',
        'runs/source_training_v2', 'runs/source_classical_v2', 'runs/meta',
        'runs/internal_validation_once')
CLAIMS = ('runs/training_campaign_v1_claim.json', 'runs/controlled_pilot_job_claim.json',
          'runs/internal_validation_once_claim.json')
ARMS = ('BASE4_PLUS_YANG', 'BASE4_PLUS_DREYER')
EXT = tuple(f'runs/{kind}/{arm}' for kind in ('source_extensions', 'source_extension_downstream') for arm in ARMS)
FIELDS = ('path', 'sha256', 'bytes', 'role', 'dataset')
BASE = '.paper3-hpc/handoffs'
R2_NAMES = ('R2_ENDPOINT', 'R2_BUCKET', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY')
NO_FIT = {'training_dispatched': False, 'scientific_completion_claimed': False, 'GPU_actions': 0,
          'credentials_recorded': False}


class HandoffError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def need(value, code):
    if not value:
        raise HandoffError(code)


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def safe_root(value):
    value = Path(os.path.abspath(value))
    need(not any(p.is_symlink() for p in (value, *value.parents)), 'handoff_symlink_forbidden')
    need(value.is_dir(), 'handoff_project_root_missing')
    return value


def local(root, relative):
    root = Path(os.path.abspath(root))
    need(not any(p.is_symlink() for p in (root, *root.parents)), 'handoff_symlink_forbidden')
    need(isinstance(relative, str) and relative and '\\' not in relative
         and not any(ord(c) < 32 for c in relative), 'unsafe_handoff_path')
    parts = PurePosixPath(relative)
    need(not parts.is_absolute() and parts.as_posix() == relative
         and all(p not in {'', '.', '..'} for p in relative.split('/')), 'unsafe_handoff_path')
    target = root
    for part in parts.parts:
        target /= part
        need(not target.is_symlink(), 'handoff_symlink_forbidden')
    return target


def hash_file(path):
    need(path.is_file() and not path.is_symlink(), 'handoff_regular_file_required')
    before = path.stat()
    h, n = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block); n += len(block)
        after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(identity(before) == identity(after) == identity(path.stat()), 'handoff_source_changed_during_read')
    return h.hexdigest(), n


def write_json(path, value, *, immutable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json_bytes(value)
    if immutable and path.exists():
        need(path.read_bytes() == raw, 'handoff_immutable_metadata_changed')
        return
    fd, name = tempfile.mkstemp(prefix='.handoff-', suffix='.partial', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        if immutable:
            os.link(name, path)
        else:
            os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def locks(root, *, producer=False, extension=False):
    paths = [] if extension else ['.paper3-hpc/handoff.lock']
    if producer:
        paths += ['.paper3-deployment/owner.lock', 'runs/.training_campaign_v1.lock']
    if extension:
        for prefix in EXT + ('runs/source_extensions/claims',):
            base = local(root, prefix)
            if base.exists():
                for path in sorted(base.rglob('*.lock')):
                    local(root, path.relative_to(root).as_posix())
                    if path.is_file():
                        paths.append(path.relative_to(root).as_posix())
    handles = []
    try:
        for relative in sorted(set(paths)):
            path = local(root, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600), 'a')
            handles.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise HandoffError('active_controller_or_handoff_preserved_no_snapshot') from None
        yield
    finally:
        for handle in reversed(handles):
            handle.close()


def original_apis():
    # Imports perform no dispatch. The root's installed scientific environment
    # is reused; no portable interpreter, pip, Docker or CUDA check is invoked.
    from tools import bootstrap_data_plan as bootstrap, r2_backup, r2_restore
    from p3work import training_campaign as campaign
    return bootstrap, campaign, r2_backup, r2_restore


def owned(row):
    need(isinstance(row, dict) and set(row) == set(FIELDS), 'exact_owned_result_descriptor_required')
    path = PurePosixPath(row['path'])
    main = row['path'] in CLAIMS or any(path.is_relative_to(prefix) for prefix in MAIN)
    extension = any(path.is_relative_to(prefix) for prefix in EXT) or (
        path.parent.as_posix() == 'runs/source_extensions/claims'
        and any(path.name.startswith(arm + '-seed-') for arm in ARMS))
    main_collection = re.fullmatch(r'\.paper3-hpc/producer_manifests/[a-f0-9]{64}\.json', row['path'])
    if main_collection:
        need(path.stem == row['sha256'], 'producer_collection_content_address_required')
    extension_collection = re.fullmatch(
        r'audit/deployment_result_snapshots/[a-f0-9]{64}_source_extensions\.json', row['path'])
    need(path.suffix in {'.json', '.npy', '.pt'} and row['role'] == 'result'
         and ((row['dataset'] == 'Paper3_mainline' and (main or main_collection))
              or (row['dataset'] == 'Paper3SourceExtension' and (extension or extension_collection))),
         'finite_original_result_scope_required')
    need(type(row['bytes']) is int and row['bytes'] >= 0
         and isinstance(row['sha256'], str) and re.fullmatch(r'[a-f0-9]{64}', row['sha256']),
         'owned_result_integrity_fields_required')


def producer_files(root, apis):
    bootstrap, campaign, _, _ = apis
    # The original writer excludes live progress and overwrites its manifest.
    # Collect the exact same original scope without changing either live file:
    # an older manifest may still be referenced by a paused campaign's state.
    main = campaign.collect_backup_manifest(root)
    need(main.get('schema_version') == 1 and isinstance(main.get('files'), list), 'original_campaign_manifest_required')
    files = list(main['files'])
    producers = []
    progress_path = local(root, 'runs/training_campaign_v1/campaign_progress.json')
    old_manifest = local(root, 'runs/training_campaign_v1/backup_manifest.json')
    if progress_path.exists():
        progress = json.loads(progress_path.read_bytes())
        need(isinstance(progress, dict), 'actual_campaign_progress_object_required')
        reference = progress.get('backup_manifest')
        if reference is not None:
            need(isinstance(reference, dict)
                 and reference.get('path') == 'runs/training_campaign_v1/backup_manifest.json'
                 and old_manifest.is_file(), 'live_progress_original_manifest_missing_preserve_checkpoint')
            sha, size = hash_file(old_manifest)
            need(reference.get('sha256') == sha and reference.get('bytes', size) == size,
                 'live_progress_original_manifest_sha_mismatch_preserve_checkpoint')
        sha, size = hash_file(progress_path)
        files.append(dict(path=progress_path.relative_to(root).as_posix(), sha256=sha, bytes=size,
                          role='result', dataset='Paper3_mainline'))
    if files:
        raw = json_bytes(main)
        relative = '.paper3-hpc/producer_manifests/' + hashlib.sha256(raw).hexdigest() + '.json'
        target = local(root, relative)
        write_json(target, main, immutable=True)
        producers.append(dict(path=relative, sha256=hash_file(target)[0], bytes=len(raw)))
        if old_manifest.exists():
            old = json.loads(old_manifest.read_bytes())
            need(isinstance(old, dict) and old.get('schema_version') == 1 and isinstance(old.get('files'), list),
                 'original_campaign_manifest_required')
            sha, size = hash_file(old_manifest)
            producers.append(dict(path=old_manifest.relative_to(root).as_posix(), sha256=sha, bytes=size))
        files += [dict(**row, role='result', dataset='Paper3_mainline') for row in producers]
    try:
        extension = bootstrap.snapshot_extension_results(root, execute=True)
    except bootstrap.BootstrapError as exc:
        if str(exc) != 'no_actual_owned_extension_outputs_to_snapshot':
            raise
        extension = None
    if extension is not None:
        ref = extension['manifest']
        need(re.fullmatch(r'audit/deployment_result_snapshots/[a-f0-9]{64}_source_extensions\.json', ref['path']),
             'original_extension_manifest_required')
        target = local(root, ref['path'])
        need(hash_file(target) == (ref['sha256'], ref['bytes']), 'producer_manifest_sha_mismatch')
        ext = json.loads(target.read_bytes())
        need(ext.get('schema_version') == 1 and ext.get('operation') == 'paper3_owned_extension_result_snapshot',
             'original_extension_manifest_required')
        need(Path(ref['path']).name == digest(ext) + '_source_extensions.json', 'original_extension_manifest_content_id_required')
        files += ext['files']; producers.append(ref)
        files.append(dict(**ref, role='result', dataset='Paper3SourceExtension'))
        # Original extension producers intentionally omit their local backup
        # manifests. Preserve those real documents too, without changing them.
        for prefix in EXT:
            base = local(root, prefix)
            if not base.exists():
                continue
            for target in sorted(base.rglob('backup_manifest.json')):
                relative = target.relative_to(root).as_posix()
                target = local(root, relative)
                value = json.loads(target.read_bytes())
                need(isinstance(value, dict) and value.get('schema_version') == 1 and isinstance(value.get('files'), list),
                     'original_extension_local_manifest_required')
                sha, size = hash_file(target)
                ref = dict(path=relative, sha256=sha, bytes=size)
                producers.append(ref)
                files.append(dict(**ref, role='result', dataset='Paper3SourceExtension'))
    seen = set()
    for row in files:
        owned(row); need(row['path'] not in seen, 'duplicate_owned_result_path'); seen.add(row['path'])
        need(hash_file(local(root, row['path'])) == (row['sha256'], row['bytes']), 'source_result_sha_mismatch')
    return sorted(files, key=lambda row: row['path']), producers


def validate_queue(root, queue):
    need(re.fullmatch(r'[a-f0-9]{64}', queue.name), 'exact_handoff_queue_id_required')
    metadata = json.loads(local(root, f'{BASE}/{queue.name}/queue.json').read_bytes())
    manifest_path = local(root, f'{BASE}/{queue.name}/manifest.json')
    manifest = json.loads(manifest_path.read_bytes())
    need(metadata.get('schema_version') == 'paper3-hpc-offline-handoff-v1'
         and metadata.get('queue_id') == queue.name
         and metadata.get('manifest_sha256') == hash_file(manifest_path)[0]
         and manifest.get('schema_version') == 1 and isinstance(manifest.get('files'), list)
         and manifest['files'], 'handoff_queue_identity_invalid')
    need(digest({'files': manifest['files'], 'producers': metadata['producer_manifests']}) == queue.name,
         'handoff_queue_content_id_mismatch')
    payload_refs = {(row.get('path'), row.get('sha256'), row.get('bytes')) for row in manifest['files']}
    for producer in metadata['producer_manifests']:
        need(isinstance(producer, dict) and set(producer) == {'path', 'sha256', 'bytes'}
             and tuple(producer[k] for k in ('path', 'sha256', 'bytes')) in payload_refs,
             'producer_metadata_payload_missing')
    for row in manifest['files']:
        owned(row)
        need(hash_file(local(queue / 'payload', row['path'])) == (row['sha256'], row['bytes']),
             'handoff_snapshot_sha_mismatch_originals_preserved')
    return metadata, manifest


def snapshot(root, *, label='manual', job_exit_code=None, apis=None):
    """Copy actual finite producer output bytes under locks, with no network."""
    root = safe_root(root)
    need(re.fullmatch(r'[A-Za-z0-9_-]{1,64}', label), 'invalid_handoff_stage_label')
    need(job_exit_code is None or type(job_exit_code) is int, 'integer_job_exit_code_required')
    apis = apis or original_apis()
    with locks(root, producer=True):
        files, producers = producer_files(root, apis)
        with locks(root, extension=True):
            return _snapshot_locked(root, files, producers, label, job_exit_code)


def _snapshot_locked(root, files, producers, label, job_exit_code):
    if not files:
        return dict(NO_FIT, status='NO_ACTUAL_OWNED_RESULTS_TO_SNAPSHOT', uploaded=False, file_count=0)
    manifest = {'schema_version': 1, 'files': files}
    queue_id = digest({'files': files, 'producers': producers})
    parent = local(root, BASE); parent.mkdir(parents=True, exist_ok=True)
    queue = local(root, BASE + '/' + queue_id)
    if queue.exists():
        validate_queue(root, queue)
    else:
        need(shutil.disk_usage(parent).free >= sum(row['bytes'] for row in files) + 1024 * 1024,
             'insufficient_snapshot_space_original_checkpoints_preserved')
        staging = Path(tempfile.mkdtemp(prefix='.snapshot-', dir=parent))
        try:
            for row in files:
                source = local(root, row['path']); target = local(staging / 'payload', row['path'])
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open('rb') as src, target.open('xb') as out:
                    shutil.copyfileobj(src, out, 1024 * 1024); out.flush(); os.fsync(out.fileno())
                need(hash_file(source) == hash_file(target) == (row['sha256'], row['bytes']),
                     'snapshot_copy_integrity_failed_originals_preserved')
            write_json(staging / 'manifest.json', manifest, immutable=True)
            metadata = {'schema_version': 'paper3-hpc-offline-handoff-v1', 'queue_id': queue_id,
                        'created_at_utc': utc(), 'manifest_sha256': hash_file(staging / 'manifest.json')[0],
                        'producer_manifests': producers, **NO_FIT,
                        'created_status': 'PENDING_EXTERNAL_BACKUP_NOT_UPLOADED'}
            write_json(staging / 'queue.json', metadata, immutable=True)
            need(not queue.exists(), 'handoff_queue_appeared_concurrently')
            os.rename(staging, queue)
        except Exception as exc:
            write_json(staging / 'failed_local_snapshot.json', dict(NO_FIT, status='LOCAL_SNAPSHOT_INCOMPLETE_ORIGINALS_PRESERVED',
                       error_code=exc.code if isinstance(exc, HandoffError) else type(exc).__name__))
            raise
    event = dict(NO_FIT, status='PENDING_EXTERNAL_BACKUP_NOT_UPLOADED', uploaded=False,
                 queue_id=queue_id, label=label, job_exit_code=job_exit_code,
                 file_count=len(files), payload_bytes=sum(row['bytes'] for row in files),
                 created_at_utc=utc(), queue_path=f'{BASE}/{queue_id}')
    write_json(queue / 'snapshot_events' / (uuid.uuid4().hex + '.json'), event, immutable=True)
    if not (queue / 'status.json').exists():
        write_json(queue / 'status.json', event)
    return event



def hidden(label):
    need(sys.stdin.isatty(), 'interactive_hidden_input_required')
    with warnings.catch_warnings():
        warnings.simplefilter('error', getpass.GetPassWarning)
        try:
            value = getpass.getpass(label + ': ').strip()
        except (EOFError, getpass.GetPassWarning):
            raise HandoffError('interactive_hidden_input_required') from None
    need(value and not any(ord(c) < 32 or ord(c) == 127 for c in value), 'invalid_hidden_credential')
    return value


def r2_client(*, prompt=False, metadata_probe=False):
    need(type(metadata_probe) is bool, 'boolean_metadata_probe_flag_required')
    credentials = {name: os.environ.get(name) for name in R2_NAMES}
    if not all(credentials.values()):
        need(prompt, 'existing_r2_configuration_required_on_network_allowed_node')
        credentials = {name: hidden(name) for name in R2_NAMES}
    from urllib.parse import urlsplit
    parsed = urlsplit(credentials['R2_ENDPOINT'])
    need(parsed.scheme == 'https' and (parsed.hostname or '').endswith('.r2.cloudflarestorage.com')
         and parsed.username is None and parsed.password is None and parsed.port in {None, 443}
         and parsed.path in {'', '/'} and not parsed.query and not parsed.fragment,
         'r2_https_account_endpoint_required')
    for name in {'boto3', 'botocore', 's3transfer', 'urllib3'} | {
            key for key in logging.Logger.manager.loggerDict if key.startswith(('boto3.', 'botocore.', 's3transfer.', 'urllib3.'))}:
        logger = logging.getLogger(name); logger.disabled = True; logger.propagate = False
    try:
        import boto3
        from botocore.config import Config
        client = boto3.client('s3', endpoint_url=credentials['R2_ENDPOINT'], region_name='auto',
                              aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'],
                              aws_secret_access_key=credentials['AWS_SECRET_ACCESS_KEY'],
                              config=Config(connect_timeout=5 if metadata_probe else 15,
                                            read_timeout=10 if metadata_probe else 120,
                                            retries={'max_attempts': 0 if metadata_probe else 2,
                                                     'mode': 'standard'}))
    except ImportError:
        raise HandoffError('existing_environment_missing_minimal_r2_sdk') from None
    bucket = credentials['R2_BUCKET']; credentials.clear()
    return client, bucket


def flush(root, *, queue_id=None, client=None, bucket=None, prompt=False, apis=None):
    """Only upload immutable handoffs; original full GET proof remains required."""
    root = safe_root(root); apis = apis or original_apis()
    _, _, backup, restore = apis
    with locks(root):
        parent = local(root, BASE)
        if queue_id is not None:
            need(re.fullmatch(r'[a-f0-9]{64}', queue_id), 'exact_handoff_queue_id_required')
            queues = [local(root, BASE + '/' + queue_id)]
        else:
            queues = sorted(p for p in parent.iterdir() if p.is_dir() and re.fullmatch(r'[a-f0-9]{64}', p.name)) if parent.exists() else []
        if not queues:
            return dict(NO_FIT, status='NO_PENDING_HANDOFFS', objects_verified=0, uploaded=False)
        if client is None:
            client, bucket = r2_client(prompt=prompt)
        need(bucket, 'existing_r2_bucket_required')
        results = []
        for queue in queues:
            metadata, manifest = validate_queue(root, queue)
            attempt = queue / 'attempts' / uuid.uuid4().hex; attempt.mkdir(parents=True)
            receipt = attempt / 'R2_receipt.json'
            prefix = 'paper3/hpc-training-results/' + queue.name + '/'
            try:
                value = backup.run_backup(manifest_path=queue / 'manifest.json', root=queue / 'payload',
                                          prefix=prefix, receipt_path=receipt, execute=True, client=client, bucket=bucket)
                need(value.get('status') == 'verified', 'r2_backup_full_get_failed_attempt_retained')
                verified, receipt_sha = restore.load_verified_receipt(receipt)
                need(len(verified) == len(manifest['files'])
                     and {tuple(row[k] for k in FIELDS) for row in verified}
                     == {tuple(row[k] for k in FIELDS) for row in manifest['files']}
                     and all(row['key'] == prefix + row['path'] for row in verified),
                     'exact_original_full_get_receipt_required')
                result = dict(NO_FIT, status='R2_PUT_AND_COMPLETE_GET_SHA256_VERIFIED', uploaded=True,
                              queue_id=queue.name, objects_verified=len(verified), verified_at_utc=utc(),
                              receipt_path=receipt.relative_to(root).as_posix(), receipt_sha256=receipt_sha)
            except Exception as exc:
                result = dict(NO_FIT, status='PENDING_EXTERNAL_BACKUP_NOT_VERIFIED', uploaded=False,
                              queue_id=queue.name, attempt_path=attempt.relative_to(root).as_posix(),
                              error_code=exc.code if isinstance(exc, HandoffError) else type(exc).__name__)
            write_json(attempt / 'result.json', result, immutable=True)
            write_json(queue / 'status.json', result); results.append(result)
        complete = all(row['uploaded'] is True for row in results)
        return dict(NO_FIT, status='ALL_SELECTED_HANDOFFS_FULL_GET_VERIFIED' if complete else 'HANDOFFS_PENDING_RETRY_BACKUP_ONLY',
                    uploaded=complete, queues=results, objects_verified=sum(r.get('objects_verified', 0) for r in results))


def _git(root, env, args, runner):
    try:
        result = runner(['git', '-c', 'credential.helper=', '-c', 'http.followRedirects=false',
                         '-c', 'core.hooksPath=/dev/null', *args], cwd=root, env=env,
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, check=False, timeout=300)
    except Exception:
        raise HandoffError('github_git_operation_failed') from None
    need(result.returncode == 0, 'github_git_operation_failed')
    return result.stdout.strip()


def publish_code(root, *, repo='jackzhu119/research3', branch=None, token=None, prompt=False, runner=subprocess.run):
    """Push only an existing committed HEAD. Never stage data or make a commit."""
    root = safe_root(root)
    need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_.-]{1,100}', repo), 'github_repo_invalid')
    token = token or os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if not token:
        need(prompt, 'existing_github_token_or_hidden_input_required')
        token = hidden('GitHub Token (Contents read/write)')
    need(all(33 <= ord(c) <= 126 for c in token) and len(token) <= 4096, 'github_token_invalid')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GIT_', 'AWS_', 'R2_', 'GH_', 'GITHUB_'))}
    for name in ('GIT_SSL_CAINFO', 'GIT_SSL_CAPATH', 'GIT_PROXY_SSL_CAINFO', 'GIT_PROXY_SSL_CAPATH'):
        if name in os.environ: env[name] = os.environ[name]
    env.update(GIT_TERMINAL_PROMPT='0', GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull)
    head = _git(root, env, ['rev-parse', 'HEAD'], runner)
    need(re.fullmatch(r'[a-f0-9]{40}', head), 'github_existing_commit_required')
    need(not _git(root, env, ['diff', '--name-only', 'HEAD', '--'], runner), 'github_tracked_changes_require_manual_commit')
    paths = _git(root, env, ['ls-tree', '-r', '--name-only', 'HEAD'], runner).splitlines()
    # Only the existing code/artifact tree is eligible. Untracked data is never
    # staged. Reject credentials/raw datasets/weights even if accidentally committed.
    for path in paths:
        parts = PurePosixPath(path).parts
        allowed = ((len(parts) == 1 and (path in {
                    'README.md', 'README_HPC.md', 'pyproject.toml', 'requirements-cpu.lock', 'requirements-gpu.lock',
                    'code_snapshot_manifest.json', 'release_manifest.json', 'ONLINE_RELEASE.json', 'ONLINE_VALIDATION.json',
                    'RUN_ON_RUNPOD.sh', '.gitignore', '.gitattributes'} or Path(path).suffix == '.py'))
                   or (parts and parts[0] in {'p3work', 'tools', 'tests'} and Path(path).suffix == '.py')
                   or (parts and parts[0] == 'hpc' and Path(path).suffix in {'.py', '.json', '.sbatch', '.sh', '.md'})
                   or path == 'audit/data_plan_20261020/deployment_finalize_review/prepare_finalize_plan_v2.py')
        if path == 'launcher_assets/Paper3_OneClick_20261021.zip':
            allowed = (_git(root, env, ['rev-parse', 'HEAD:' + path], runner)
                       == '06af636d8de736a378afb62b5da2c506bdd6983f')
        forbidden = (not parts or any(p in {'.aws', '.ssh', 'credentials', 'secrets'} or p.startswith('.env') for p in parts))
        need(allowed and not forbidden, 'github_data_or_credential_file_forbidden')
    branch = branch or 'research/paper3-hpc-code-' + head[:12]
    need(branch.startswith(('research/', 'paper3/')) and re.fullmatch(r'[A-Za-z0-9_./-]+', branch)
         and not any(x in branch for x in ('..', '//', '@{')) and not branch.endswith(('/', '.lock', '.')),
         'github_research_branch_required')
    with tempfile.TemporaryDirectory(prefix='paper3-hpc-git-auth-') as directory:
        helper = Path(directory) / 'askpass.py'
        helper.write_text('#!' + sys.executable + '\nimport os,sys\np=sys.argv[1].lower()\n'
                          'if "github.com" not in p:sys.exit(1)\n'
                          'if p.startswith("username"):print("x-access-token")\n'
                          'elif p.startswith("password"):print(os.environ["PAPER3_HPC_GITHUB_TOKEN"])\n'
                          'else:sys.exit(1)\n')
        helper.chmod(0o700)
        env.update(GIT_ASKPASS=str(helper), PAPER3_HPC_GITHUB_TOKEN=token)
        try:
            url = 'https://github.com/' + repo + '.git'
            def remote():
                value = _git(root, env, ['ls-remote', '--heads', url, 'refs/heads/' + branch], runner)
                if not value: return None
                fields = value.split()
                need(len(fields) == 2 and fields[1] == 'refs/heads/' + branch
                     and re.fullmatch(r'[a-f0-9]{40}', fields[0]), 'github_remote_sha_invalid')
                return fields[0]
            prior = remote(); need(prior in {None, head}, 'github_remote_branch_differs_preserved')
            if prior is None:
                _git(root, env, ['push', url, 'HEAD:refs/heads/' + branch], runner)
            need(remote() == head, 'github_remote_commit_not_verified')
            return dict(NO_FIT, status='GITHUB_COMMITTED_CODE_REMOTE_SHA_VERIFIED', repository=repo,
                        branch=branch, commit=head, actual_push_performed=prior is None)
        finally:
            env.pop('PAPER3_HPC_GITHUB_TOKEN', None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--snapshot', action='store_true', help='offline immutable result handoff; no upload')
    mode.add_argument('--flush', action='store_true', help='explicit network-node PUT and complete GET verification')
    parser.add_argument('--label', default='manual'); parser.add_argument('--job-exit-code', type=int)
    parser.add_argument('--queue-id'); parser.add_argument('--publish-code', action='store_true')
    parser.add_argument('--repo', default='jackzhu119/research3'); parser.add_argument('--branch')
    args = parser.parse_args(argv)
    try:
        need(not args.publish_code or args.flush, 'code_publication_only_with_explicit_flush')
        if args.snapshot:
            result = snapshot(args.root, label=args.label, job_exit_code=args.job_exit_code)
        elif args.flush:
            result = flush(args.root, queue_id=args.queue_id, prompt=True)
            if args.publish_code:
                try:
                    result['github'] = publish_code(args.root, repo=args.repo, branch=args.branch, prompt=True)
                except Exception as exc:
                    result['github'] = dict(status='GITHUB_PUBLICATION_PENDING',
                                            error_code=exc.code if isinstance(exc, HandoffError) else type(exc).__name__)
            result['selected_code_publication_verified'] = not args.publish_code or result['github'].get('status') == 'GITHUB_COMMITTED_CODE_REMOTE_SHA_VERIFIED'
        else:
            result = dict(NO_FIT, status='READONLY_HANDOFF_PLAN', uploaded=False,
                          instruction='Use --snapshot after controller release; use --flush only on a network-permitted node.')
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get('status') not in {'HANDOFFS_PENDING_RETRY_BACKUP_ONLY'} and result.get('selected_code_publication_verified', True) else 2
    except Exception as exc:
        print(json.dumps(dict(NO_FIT, status='HANDOFF_NOT_COMPLETED_ORIGINALS_PRESERVED', uploaded=False,
                   error_code=exc.code if isinstance(exc, HandoffError) else type(exc).__name__)))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
