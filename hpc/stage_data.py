#!/usr/bin/env python3
"""Stage the frozen source/development data without installing or training.

``check`` and ``materialize`` never use the network. ``fetch`` is an explicit
network-host operation, using the original profile-bound R2 restore code and
its exact receipt keys. Frozen software is verified but never overlaid on the
current checkout. No sealed query, model fit, dependency or server is started.
"""
from __future__ import annotations

import argparse
import fcntl
import errno
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import uuid

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

import paper3_transport as transport
from tools import bootstrap_data_plan as bootstrap
from tools import r2_restore
from p3work import source_extension_io

SCHEMA = 'paper3-hpc-data-stage-v1'
RELEASE_SHA256 = '2f48f4baf5a1069f21f324f9a43e63dea9e87a2f19f5167ce6c9bdee14177c60'
METADATA_ROOTS = frozenset({'audit', 'catalog', 'data_catalog', 'manifests',
                            'contracts', 'plans', 'protocol', 'status', 'provenance'})
CACHE_ROOTS = frozenset({'source_v2', 'stieger_v2', 'source_extensions_20261019',
                         'source_c3czc4_v1'})
DATA_SUFFIXES = frozenset({'.npy', '.json', '.jsonl', '.tsv', '.csv', '.txt'})


class DataStageError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def need(condition, code):
    if not condition:
        raise DataStageError(code)


def load_release():
    path = REPOSITORY / 'hpc/data_release.json'
    need(transport.hash_file(path)[0] == RELEASE_SHA256, 'fixed_data_release_descriptor_changed')
    value = json.loads(path.read_bytes())
    need(value.get('schema_version') == 'paper3-hpc-fixed-data-release-v1', 'fixed_data_release_required')
    need(set(value.get('objects', {})) == {'main_archive', 'main_inputs', 'control_archive',
                                         'control_inputs', 'ready'}, 'exact_five_release_objects_required')
    return value


def _allowed(path):
    """Only frozen metadata and the original four numeric-cache namespaces."""
    relative = transport._relative(path)
    # Reuse the frozen bootstrap credential/path guard; no data access occurs.
    bootstrap.checked_path(REPOSITORY, path)
    if relative.parts[0] in METADATA_ROOTS:
        return relative.suffix.lower() not in {'.edf', '.bdf', '.gdf', '.mat', '.set',
            '.fdt', '.cnt', '.eeg', '.dat', '.zip', '.7z', '.tar', '.gz', '.npz',
            '.pt', '.pth', '.ckpt', '.pkl', '.pickle'}
    return (len(relative.parts) >= 3 and relative.parts[0] == 'cache'
            and relative.parts[1] in CACHE_ROOTS and relative.suffix.lower() in DATA_SUFFIXES)


def _file(root, relative, *, create=False):
    # Missing parent directories are an absent payload, not a corrupt one.
    # checked_path rejects every symlink component even for absent leaves.
    target = bootstrap.checked_path(root, relative)
    return transport._file(root, relative, create_parents=True) if create else target


def _verify(root, row):
    path = _file(root, row['path'])
    if not path.exists():
        raise FileNotFoundError(errno.ENOENT, 'required_frozen_file_missing', str(path))
    transport._matches(path, row)
    return path


def _atomic_json(root, relative, value):
    target = _file(root, relative, create=True)
    # Only our invocation receipt may be replaced, never a scientific input.
    output, temporary = transport._temporary(target.parent)
    try:
        with output:
            output.write((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode())
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_stream(root, row, stream):
    """Whole-body SHA and no-overwrite commit, matching the original IO rules."""
    target = _file(root, row['path'], create=True)
    if target.exists():
        transport._matches(target, row)
        return 'already_present_verified'
    output, temporary = transport._temporary(target.parent)
    digest, size = hashlib.sha256(), 0
    try:
        with output:
            while block := stream.read(transport.CHUNK):
                size += len(block)
                need(size <= row['bytes'], 'materialized_body_too_long')
                digest.update(block)
                output.write(block)
            need((digest.hexdigest(), size) == (row['sha256'], row['bytes']),
                 'materialized_body_SHA_or_size_mismatch')
            output.flush()
            os.fsync(output.fileno())
        try:
            transport._commit(root, row['path'], temporary, 0o644)
        except transport.TransportError as exc:
            if exc.code != 'ATOMIC_TARGET_ALREADY_EXISTS':
                raise
            transport._matches(target, row)
        return 'materialized_SHA_verified'
    finally:
        temporary.unlink(missing_ok=True)


def _package_rows(archive_root, release, name):
    row = release['objects'][name + '_inputs']
    path = _verify(archive_root, row)
    top = 'paper3_data_plan_' + release['closure_sha256'][:16]
    value, rows = transport._package_inputs(path, top)
    need(value.get('closure_sha256') == release['closure_sha256']
         and value.get('bootstrap_index') == release['bootstrap_index'], 'package_closure_or_index_mismatch')
    return top, rows


def _materialize_archive(root, archive_root, release, name):
    archive = _verify(archive_root, release['objects'][name + '_archive'])
    top, rows = _package_rows(archive_root, release, name)
    counts = {'materialized': 0, 'already_present': 0, 'excluded_software_or_other': 0,
              'archive_members': len(rows)}
    with tarfile.open(archive, 'r:gz') as package:
        members = package.getmembers()
        expected = {top + '/' + relative: row for relative, row in rows.items()}
        names = [member.name for member in members]
        need(len(names) == len(set(names)) and set(names) == set(expected),
             'archive_exact_member_set_mismatch')
        for member in members:
            row = expected[member.name]
            mode = 0o755 if Path(row['path']).suffix in {'.py', '.sh'} else 0o644
            need(member.isfile() and member.type in {tarfile.REGTYPE, tarfile.AREGTYPE}
                 and member.size == row['bytes'] and member.mode == mode
                 and member.uid == member.gid == member.mtime == 0
                 and not member.uname and not member.gname, 'archive_regular_frozen_header_required')
            transport._relative(member.name)
        while trailer := package.fileobj.read(transport.CHUNK):
            need(not any(trailer), 'archive_nonzero_trailing_data')
        for member in members:
            row = expected[member.name]
            if not _allowed(row['path']):
                counts['excluded_software_or_other'] += 1
                continue
            with package.extractfile(member) as stream:
                result = _copy_stream(root, row, stream)
            key = 'materialized' if result == 'materialized_SHA_verified' else 'already_present'
            counts[key] += 1
    # The complete compressed file was SHA-verified before opening; repeat after
    # materialization to reject a concurrent archive replacement.
    _verify(archive_root, release['objects'][name + '_archive'])
    return counts


def _load_index(root, release):
    path = _verify(root, release['bootstrap_index'])
    value = json.loads(path.read_bytes())
    need(value.get('schema_version') == bootstrap.SCHEMA
         and value.get('closure_sha256') == release['closure_sha256']
         and bootstrap.digest({k: v for k, v in value.items() if k != 'index_sha256'})
         == value.get('index_sha256') and value.get('raw_payloads_selected') == 0
         and value.get('external_query_access_allowed') is False
         and value.get('user_manual_GPU_boot_required') is True,
         'fixed_index_identity_or_scope_invalid')
    need(value.get('runtime') == {'path': bootstrap.RUNTIME, 'sha256': bootstrap.RUNTIME_SHA,
                                  'bytes': bootstrap.RUNTIME_BYTES}
         and value.get('thread_environment') == {key: '1' for key in bootstrap.THREAD_VARIABLES},
         'frozen_runtime_descriptor_or_thread_profile_changed')
    by_path = {}
    for row in value.get('files', []):
        need(set(row) == set(bootstrap.ENTRY_FIELDS) | {'io_profile'}
             and row['io_profile'] in {'baseline', 'source_extension'}, 'exact_profiled_file_required')
        transport._descriptor(row, exact=False)
        bootstrap.safe_payload(row)
        need(row['path'] not in by_path, 'duplicate_frozen_file')
        by_path[row['path']] = row
    need(bool(by_path), 'nonempty_frozen_index_required')
    covered = set()
    loaders = {'baseline': r2_restore, 'source_extension': source_extension_io._modules()[1]}
    for group in value.get('restore_groups', []):
        need(group.get('io_profile') in loaders and group.get('paths')
             and len(group['paths']) == len(set(group['paths'])), 'exact_restore_group_required')
        receipt = _verify(root, group['receipt'])
        entries, receipt_sha = loaders[group['io_profile']].load_verified_receipt(receipt)
        need(receipt_sha == group['receipt']['sha256'], 'original_restore_receipt_SHA_changed')
        received = {entry['path']: entry for entry in entries}
        for relative in group['paths']:
            need(relative in by_path and relative not in covered
                 and by_path[relative]['io_profile'] == group['io_profile']
                 and relative in received
                 and all(received[relative][key] == by_path[relative][key]
                         for key in bootstrap.ENTRY_FIELDS), 'original_restore_selection_binding_invalid')
            covered.add(relative)
    need(covered == set(by_path), 'every_frozen_file_requires_exact_original_restore_group')
    for key in ('runtime_contract', 'deploy_manifest'):
        _verify(root, value[key])
    names = []
    for row in value.get('bootstrap_wheels', []):
        transport._descriptor(row)
        need(row['path'] in by_path and all(row[k] == by_path[row['path']][k] for k in row),
             'frozen_SDK_descriptor_outside_index')
        names.append(PurePosixPath(row['path']).name.split('-')[0])
    need(sorted(names) == sorted(bootstrap.SDK_NAMES), 'frozen_exact_SDK_descriptor_set_required')
    need(set(value.get('extension_manifests', {})) == set(bootstrap.ARMS)
         and all(path in by_path for path in value['extension_manifests'].values()),
         'both_original_extension_manifests_required')
    # Deliberately do not verify/install the unused portable runtime or wheels;
    # their exact descriptors remain frozen above. Current software is separate.
    return value


def load_index(root):
    """Return the original, fully bound index (no altered scientific fields)."""
    return _load_index(transport._root(root), load_release())


def _inspect(root, index):
    selected = [row for row in index['files'] if _allowed(row['path'])]
    def inspect_one(row):
        target = _file(root, row['path'])
        if target.exists():
            transport._matches(target, row)
            return None
        return {key: row[key] for key in bootstrap.ENTRY_FIELDS + ('io_profile',)}
    # Parallel independent SHA/IO only; scientific BLAS/fit threads stay frozen.
    try:
        cpus = max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    except ValueError:
        cpus = 1
    workers = min(8, cpus)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        missing = [row for row in executor.map(inspect_one, selected) if row is not None]
    present = len(selected) - len(missing)
    paths = {row['path'] for row in missing}
    groups = []
    for ordinal, group in enumerate(index['restore_groups']):
        chosen = [path for path in group['paths'] if path in paths]
        if chosen:
            groups.append({'ordinal': ordinal, 'io_profile': group['io_profile'],
                           'receipt': group['receipt'], 'paths': chosen})
    return {'status': 'blocked' if missing else 'ready', 'index_sha256': index['index_sha256'],
            'frozen_data_files': len(selected), 'locally_SHA_verified': present,
            'independent_SHA_IO_workers': workers,
            'missing_files': missing, 'missing_bytes': sum(row['bytes'] for row in missing),
            'missing_cache_files': sum(row['path'].startswith('cache/') for row in missing),
            'missing_cache_bytes': sum(row['bytes'] for row in missing if row['path'].startswith('cache/')),
            'restore_groups': groups,
            'unused_software_files_not_materialized': len(index['files']) - len(selected),
            'minimal_gap': ('Run explicit network-host fetch or transfer the exact missing files with their '
                            'original receipt; rerun offline check.' if missing else None)}


def _run(root, archive_root, release, *, mode, execute, network_host, client, bucket, prompt, persist):
    need(mode in {'check', 'materialize', 'fetch'}, 'unknown_data_stage_mode')
    if mode == 'fetch':
        need(execute and network_host, 'fetch_requires_execute_and_network_host')
        need(not os.environ.get('SLURM_JOB_ID'), 'fetch_is_separate_from_offline_compute_job')
        if client is None:
            from hpc.handoff import r2_client
            client, configured_bucket = r2_client(prompt=prompt)
            bucket = bucket or configured_bucket
        bucket = bucket or os.environ.get('R2_BUCKET')
        need(bool(bucket), 'existing_R2_bucket_required')
    report = {'schema_version': SCHEMA, 'operation': mode, 'status': 'running',
              'root': str(root), 'archive_root': str(archive_root),
              'closure_sha256': release['closure_sha256'], 'credentials_recorded': False,
              'external_query_accessed': False, 'scientific_fits': 0, 'GPU_checked': False,
              'dependency_installation_performed': False, 'source_code_overlaid': False,
              'release_object_GETs': [], 'group_restore_receipts': []}
    persist(report)
    if mode == 'fetch':
        for row in release['objects'].values():
            report['release_object_GETs'].append(transport.fetch_object(client, bucket, row, archive_root))
            persist(report)
    if mode in {'materialize', 'fetch'}:
        report['archives'] = {}
        for name in ('main', 'control'):
            report['archives'][name] = _materialize_archive(root, archive_root, release, name)
            persist(report)
    index = _load_index(root, release)
    inspection = _inspect(root, index)
    if mode == 'fetch' and inspection['restore_groups']:
        for group in inspection['restore_groups']:
            loader = r2_restore if group['io_profile'] == 'baseline' else source_extension_io
            relative = report['attempt'] + '/restores/' + str(group['ordinal']).zfill(5) + '.json'
            result = loader.run_restore(backup_receipt_path=root / group['receipt']['path'], root=root,
                report_path=_file(root, relative, create=True), paths=group['paths'], execute=True,
                client=client, bucket=bucket)
            report['group_restore_receipts'].append({'ordinal': group['ordinal'], 'path': relative,
                'status': result.get('status'), 'objects': len(result.get('objects', [])),
                'fresh_full_GETs': sum(row.get('status') == 'restored_verified'
                    and row.get('get_attempted') is True and row.get('complete_body_read') is True
                    for row in result.get('objects', []))})
            persist(report)
            need(result.get('status') == 'restored_verified', 'original_profile_restore_failed_see_receipt')
        inspection = _inspect(root, index)
    report.update(inspection)
    report['ready_scope'] = 'frozen_W3_source_development_and_internal_cache_staging_only_external_query_forbidden'
    persist(report)
    return report


def prepare_data(root, *, archive_root=None, mode='check', execute=False, network_host=False,
                 client=None, bucket=None, prompt=False):
    """Run a resumable, locked data-only stage; return ready/blocked/failed proof.

    ``archive_root`` contains the frozen descriptors' relative ``archives/`` and
    ``audit/`` paths. Default PROJECT/.hpc/packages is user-transferable. No data
    access occurs merely by importing this module.
    """
    root = transport._root(root)
    archive_root = transport._root(archive_root or root / '.hpc/packages', create=True)
    owner = _file(root, '.paper3-hpc/data/stage.lock', create=True).open('a')
    report, attempt = {}, '.paper3-hpc/data/attempt-' + uuid.uuid4().hex
    def persist(value):
        nonlocal report
        report = value
        value['attempt'] = attempt
        _atomic_json(root, attempt + '/receipt.json', value)
    try:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'schema_version': SCHEMA, 'status': 'blocked', 'error_code': 'data_stage_already_running',
                    'scientific_fits': 0, 'external_query_accessed': False}
        report = _run(root, archive_root, load_release(), mode=mode, execute=execute,
                      network_host=network_host, client=client, bucket=bucket, prompt=prompt, persist=persist)
    except Exception as exc:
        # SDK exceptions and credential values are never stringified/serialized.
        code = getattr(exc, 'code', None)
        if isinstance(exc, bootstrap.BootstrapError) and len(exc.args) == 1:
            # The frozen bootstrap emits only constant machine codes.
            code = exc.args[0]
        if not isinstance(code, str) or not re.fullmatch('[A-Za-z0-9_]+', code):
            code = 'missing_frozen_input' if isinstance(exc, FileNotFoundError) else 'data_stage_failed'
        blocked = isinstance(exc, FileNotFoundError) or code in {
            'existing_r2_configuration_required_on_network_allowed_node',
            'existing_environment_missing_minimal_r2_sdk'}
        missing = []
        if isinstance(exc, FileNotFoundError) and exc.filename:
            for base in (root, archive_root):
                try:
                    missing.append(Path(exc.filename).relative_to(base).as_posix())
                    break
                except ValueError:
                    pass
        report.update(schema_version=SCHEMA, status='blocked' if blocked else 'failed',
                      missing_inputs=missing,
                      error_code=code, credentials_recorded=False, scientific_fits=0,
                      external_query_accessed=False, GPU_checked=False,
                      minimal_gap='Supply exact frozen packages/sidecars in .hpc/packages or run explicit network-host fetch; inspect this attempt receipt.')
        persist(report)
    finally:
        owner.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'materialize', 'fetch'))
    parser.add_argument('--root', type=Path, default=REPOSITORY)
    parser.add_argument('--archive-root', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--network-host', action='store_true')
    args = parser.parse_args(argv)
    result = prepare_data(args.root, archive_root=args.archive_root, mode=args.mode,
                          execute=args.execute, network_host=args.network_host,
                          prompt=args.mode == 'fetch' and sys.stdin.isatty())
    brief = {key: result.get(key) for key in ('status', 'attempt', 'locally_SHA_verified',
        'missing_cache_files', 'missing_cache_bytes', 'error_code', 'minimal_gap') if key in result}
    print(json.dumps(brief, ensure_ascii=False))
    return 0 if result.get('status') == 'ready' else 2


if __name__ == '__main__':
    raise SystemExit(main())
