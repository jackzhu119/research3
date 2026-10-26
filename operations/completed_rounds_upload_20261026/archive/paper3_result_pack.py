"""Read completed Paper 3 scientific outputs into resumable deterministic archives.

Only standard-library modules are used. Source outputs are never modified.
This is a scientific-output backup, not a raw-dataset/shared-source-model backup.
"""
import contextlib
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile
import time
import zlib


class PackSourceChanged(RuntimeError):
    pass


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')


def _atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temp.open('wb') as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _read_json(path):
    with Path(path).open('rb') as handle:
        return json.load(handle)


def _digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _signature(st):
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns,
            stat.S_IMODE(st.st_mode)]


def _emit(callback, event, **fields):
    callback(event, **fields)


def _task_info(out, plan, callback):
    tasks = []
    signal = hashlib.sha256()
    last = time.monotonic()
    for index, task in enumerate(plan.get('tasks', [])):
        key = task.get('key')
        if not isinstance(key, str) or not key or Path(key).is_absolute() or '..' in Path(key).parts:
            raise ValueError('Invalid task key in source plan')
        folder = out / key
        result_path = folder / 'result.json'
        try:
            data = result_path.read_bytes()
            result = json.loads(data)
        except FileNotFoundError:
            data, result = b'', {}
        except (ValueError, TypeError):
            data, result = b'', {}
        status = result.get('status', 'MISSING_OR_UNREADABLE') if isinstance(result, dict) else 'UNREADABLE'
        result_sha = hashlib.sha256(data).hexdigest() if data else None
        row = {'key': key, 'family': task.get('family', task.get('arm', 'other')),
               'status': status, 'result_sha256': result_sha}
        signal.update(_json_bytes(row))
        tasks.append(row)
        if time.monotonic() - last >= 5:
            _emit(callback, 'CHECKING_COMPLETION', checked=index + 1,
                  total=len(plan['tasks']))
            last = time.monotonic()
    return tasks, signal.hexdigest()


_EXCLUDED_NAMES = {'private', 'runtime', 'logs', '__pycache__', '.git',
                   'STOP', 'github.json', 'credentials.json', 'credentials',
                   'token', 'token.txt', '.env'}


def _inventory(out, plan):
    """Select scientific output only; never follow links outside output."""
    rows, excluded, seen = [], [], set()
    task_keys = {task['key'] for task in plan['tasks']}
    roots = [(out / key, False) for key in sorted(task_keys)]
    roots += [(out / 'plan.json', True), (out / 'report.json', True),
              (out / 'reports', True)]
    roots += [(path, True) for path in sorted(out.glob('*.csv'))]
    for base, summary in roots:
        if base.is_symlink():
            excluded.append({'path': base.relative_to(out).as_posix(), 'reason': 'symlink'})
            continue
        if not base.exists():
            continue
        candidates = []
        if base.is_dir():
            for directory, dirs, files in os.walk(base, followlinks=False):
                directory = Path(directory)
                kept = []
                for name in sorted(dirs):
                    path = directory / name
                    if name in _EXCLUDED_NAMES or path.is_symlink():
                        excluded.append({'path': path.relative_to(out).as_posix(),
                                         'reason': 'symlink' if path.is_symlink() else 'operational_or_private'})
                    else:
                        kept.append(name)
                dirs[:] = kept
                candidates.extend(directory / name for name in sorted(files))
        else:
            candidates = [base]
        for path in candidates:
            relative = path.relative_to(out).as_posix()
            if relative in seen:
                continue
            seen.add(relative)
            if path.name in _EXCLUDED_NAMES or path.is_symlink():
                excluded.append({'path': relative, 'reason': 'symlink' if path.is_symlink() else 'operational_or_private'})
                continue
            st = path.lstat()
            if not stat.S_ISREG(st.st_mode):
                excluded.append({'path': relative, 'reason': 'not_regular_file'})
                continue
            rows.append({'path': relative, 'bytes': st.st_size,
                         'signature': _signature(st), 'summary': summary})
    return sorted(rows, key=lambda row: row['path']), sorted(excluded, key=lambda row: row['path'])


class _PartWriter:
    def __init__(self, directory, round_number, part_bytes):
        self.directory = Path(directory)
        self.round_number = round_number
        self.part_bytes = part_bytes
        self.parts = []
        self.handle = None
        self.size = self.position = 0
        self.hasher = None

    def _open(self):
        name = 'round-%02d.tar.gz.part-%05d' % (self.round_number, len(self.parts) + 1)
        self.name = name
        self.temp = self.directory / (name + '.tmp')
        self.handle = self.temp.open('wb')
        self.size = 0
        self.hasher = hashlib.sha256()

    def _finish(self):
        if self.handle is None:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        target = self.directory / self.name
        os.replace(self.temp, target)
        self.parts.append({'name': self.name, 'bytes': self.size,
                           'sha256': self.hasher.hexdigest()})
        self.handle = None

    def write(self, data):
        view = memoryview(data)
        consumed = 0
        while consumed < len(view):
            if self.handle is None:
                self._open()
            end = min(len(view), consumed + self.part_bytes - self.size)
            chunk = view[consumed:end]
            self.handle.write(chunk)
            self.hasher.update(chunk)
            amount = len(chunk)
            consumed += amount
            self.size += amount
            self.position += amount
            if self.size == self.part_bytes:
                self._finish()
        return consumed

    def flush(self):
        if self.handle is not None:
            self.handle.flush()

    def tell(self):
        return self.position

    def close(self):
        self._finish()


class _DigestReader:
    def __init__(self, source, copy=None, capture=False):
        self.source = source
        self.copy = copy
        self.hasher = hashlib.sha256()
        self.capture = bytearray() if capture else None
        self.bytes = 0

    def read(self, size=-1):
        data = self.source.read(size)
        self.hasher.update(data)
        self.bytes += len(data)
        if self.copy is not None:
            self.copy.write(data)
        if self.capture is not None:
            if len(self.capture) + len(data) <= 4 * 1024 * 1024:
                self.capture.extend(data)
            else:
                self.capture = None
        return data


def _cached_ready(round_work, source_signature, callback):
    pointer = round_work / 'prepared.json'
    try:
        saved = _read_json(pointer)
        if saved.get('source_signature') != source_signature:
            return None
        directory = round_work / saved['snapshot_id']
        if _digest_file(directory / 'manifest.json') != saved.get('manifest_sha256'):
            return None
        manifest = _read_json(directory / 'manifest.json')
        if (manifest.get('snapshot_id') != saved['snapshot_id'] or
                sum(row['bytes'] for row in manifest['parts']) != manifest['archive_bytes']):
            return None
        for part in manifest['parts']:
            path = directory / part['name']
            if path.stat().st_size != part['bytes'] or _digest_file(path) != part['sha256']:
                return None
        for row in saved.get('summary_files', []):
            path = directory / 'git-files' / row['path']
            if path.stat().st_size != row['bytes'] or _digest_file(path) != row['sha256']:
                return None
        _emit(callback, 'LOCAL_ARCHIVE_REUSED', round=saved['round'],
              snapshot_id=saved['snapshot_id'], parts=len(manifest['parts']))
        return _ready_result(directory, manifest, saved['summary_files'], True)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _ready_result(directory, manifest, summaries, reused):
    return {'status': 'READY', 'round': manifest['round'],
            'source_output': manifest['source_output'],
            'snapshot_id': manifest['snapshot_id'],
            'total_tasks': manifest['total_tasks'], 'complete_tasks': manifest['complete_tasks'],
            'manifest_path': str(directory / 'manifest.json'), 'manifest': manifest,
            'parts': [{**row, 'path': str(directory / row['name'])} for row in manifest['parts']],
            'summary_files': [{**row, 'local_path': str(directory / 'git-files' / row['path'])}
                              for row in summaries], 'reused': reused}


def prepare_round(root, round_number, work, emit, part_bytes=32 * 1024 * 1024):
    """Return READY or PENDING; all paths returned refer to frozen work files.

    emit is called as emit(event_name, **fields). It must not print credentials.
    An incomplete/crashed packing stage is never reused as an uploadable archive.
    """
    root, work = Path(root), Path(work)
    if round_number not in (4, 5, 6, 8):
        raise ValueError('This exporter supports rounds 4, 5, 6 and 8 only')
    if not isinstance(part_bytes, int) or part_bytes < 1024:
        raise ValueError('Archive part size must be at least 1024 bytes')
    name = ('followup_20261025_r8' if round_number == 8 else 'followup_20261024_r%d' % round_number)
    out = root / 'runs' / name
    plan_path = out / 'plan.json'
    plan_data = plan_path.read_bytes()
    plan = json.loads(plan_data)
    if not isinstance(plan, dict) or not isinstance(plan.get('tasks'), list) or not plan['tasks']:
        raise ValueError('Source round task plan is missing or invalid')
    tasks, completion_digest = _task_info(out, plan, emit)
    complete = sum(row['status'] == 'COMPLETE' for row in tasks)
    if complete != len(tasks):
        return {'status': 'PENDING', 'round': round_number, 'source_output': str(out),
                'complete_tasks': complete, 'total_tasks': len(tasks),
                'pending_tasks': [row['key'] for row in tasks if row['status'] != 'COMPLETE']}
    round_work = work / ('round-%02d' % round_number)
    round_work.mkdir(parents=True, exist_ok=True)
    with (round_work / '.pack.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        inventory, excluded = _inventory(out, plan)
        source_signature = hashlib.sha256(_json_bytes({
            'plan_sha256': hashlib.sha256(plan_data).hexdigest(),
            'completion_digest': completion_digest,
            'files': inventory, 'excluded': excluded, 'part_bytes': part_bytes})).hexdigest()
        ready = _cached_ready(round_work, source_signature, emit)
        if ready:
            return ready
        for stale in round_work.glob('.packing-*'):
            if stale.is_dir() and not stale.is_symlink():
                shutil.rmtree(stale)
        stage = round_work / ('.packing-%d-%d' % (os.getpid(), time.time_ns()))
        stage.mkdir()
        summaries, files = [], []
        captured_summary_sources = {}
        by_key = {row['key']: row for row in tasks}
        writer = _PartWriter(stage, round_number, part_bytes)
        last = time.monotonic()
        packed_bytes = 0
        _emit(emit, 'PACK_STARTED', round=round_number, files=len(inventory),
              source_bytes=sum(row['bytes'] for row in inventory), part_bytes=part_bytes)
        try:
            with gzip.GzipFile(filename='', mode='wb', fileobj=writer, mtime=0, compresslevel=3) as gz:
                with tarfile.open(mode='w|', fileobj=gz, format=tarfile.PAX_FORMAT) as archive:
                    for index, row in enumerate(inventory):
                        relative = row['path']
                        path = out / relative
                        flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
                        asynchronous_summary = row['summary'] and relative != 'plan.json'
                        try:
                            fd = os.open(str(path), flags)
                        except FileNotFoundError:
                            if asynchronous_summary:
                                _emit(emit, 'SUMMARY_DISAPPEARED_BEFORE_CAPTURE', round=round_number,
                                      path=relative)
                                continue
                            raise
                        with os.fdopen(fd, 'rb') as source:
                            st = os.fstat(source.fileno())
                            if (not stat.S_ISREG(st.st_mode) or
                                    (not asynchronous_summary and _signature(st) != row['signature'])):
                                raise PackSourceChanged('Scientific output changed before packing: ' + relative)
                            if asynchronous_summary:
                                # A monitor may atomically replace reports while
                                # immutable completed task outputs are packed.
                                # Capture exactly the version opened here, not
                                # the earlier directory-inventory version.
                                row = {**row, 'bytes': st.st_size, 'signature': _signature(st)}
                            copy = None
                            if row['summary']:
                                dest = stage / 'git-files' / relative
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                copy = dest.open('wb')
                            capture = relative.endswith('/progress.json') and len(Path(relative).parts) == 2
                            reader = _DigestReader(source, copy, capture)
                            try:
                                info = archive.gettarinfo(fileobj=source, arcname=relative)
                                info.type = tarfile.REGTYPE
                                info.linkname = ''
                                info.size = st.st_size
                                info.uid = info.gid = 0
                                info.uname = info.gname = ''
                                info.mtime = 0
                                info.mode = 0o644
                                info.pax_headers = {}
                                archive.addfile(info, reader)
                            finally:
                                if copy is not None:
                                    copy.close()
                            if reader.bytes != st.st_size or _signature(os.fstat(source.fileno())) != _signature(st):
                                raise PackSourceChanged('Scientific output changed while packing: ' + relative)
                        digest = reader.hasher.hexdigest()
                        file_row = {'path': relative, 'bytes': reader.bytes, 'sha256': digest}
                        files.append(file_row)
                        if row['summary']:
                            summaries.append(file_row.copy())
                            captured_summary_sources[relative] = row
                        parts = Path(relative).parts
                        if len(parts) == 2 and parts[0] in by_key:
                            task = by_key[parts[0]]
                            if parts[1] == 'result.json' and digest != task['result_sha256']:
                                raise PackSourceChanged('Task result changed during completion check: ' + relative)
                            if capture and reader.capture is not None:
                                try:
                                    value = json.loads(reader.capture)
                                    keys = ('status', 'stage', 'outer_update', 'total_updates', 'session',
                                            'sessions_total', 'budget', 'updated_utc', 'outer_updates_completed')
                                    task['progress'] = {key: value[key] for key in keys if key in value}
                                except (ValueError, TypeError, KeyError):
                                    task['progress'] = {'status': 'UNREADABLE_PROGRESS'}
                        packed_bytes += reader.bytes
                        if time.monotonic() - last >= 5 or index == len(inventory) - 1:
                            _emit(emit, 'PACK_PROGRESS', round=round_number, files=index + 1,
                                  total_files=len(inventory), source_bytes=packed_bytes,
                                  archive_bytes=writer.position)
                            last = time.monotonic()
            writer.close()
            after, excluded_after = _inventory(out, plan)
            # Completed task artifacts and plan remain a coherent immutable
            # snapshot. Aggregate reports are independently frozen per file:
            # later report replacements/additions must not invalidate an
            # otherwise complete multi-hour task archive.
            before_map = {row['path']: row for row in inventory
                          if not row['summary'] or row['path'] == 'plan.json'}
            after_map = {row['path']: row for row in after
                         if not row['summary'] or row['path'] == 'plan.json'}
            task_keys = set(by_key)
            def task_exclusions(rows):
                return [row for row in rows if any(row['path'] == key or
                        row['path'].startswith(key + '/') for key in task_keys)]
            if (before_map.keys() != after_map.keys() or
                    task_exclusions(excluded_after) != task_exclusions(excluded)):
                raise PackSourceChanged('Scientific output inventory changed during packing')
            content = {row['path']: row for row in files}
            for relative, row in after_map.items():
                if row['signature'] != before_map[relative]['signature']:
                    if _digest_file(out / relative) != content[relative]['sha256']:
                        raise PackSourceChanged('Scientific output changed after packing: ' + relative)
            _, completion_after = _task_info(out, plan, emit)
            if completion_after != completion_digest or plan_path.read_bytes() != plan_data:
                raise PackSourceChanged('Plan or completed task receipts changed during packing')
            identity = {'schema_version': 'paper3-scientific-output-snapshot-v1',
                        'round': round_number, 'source_output': str(out),
                        'plan_sha256': hashlib.sha256(plan_data).hexdigest(),
                        'archive_layout': {'part_bytes': part_bytes, 'gzip_level': 3,
                                           'zlib_version': zlib.ZLIB_VERSION,
                                           'format': 'pax-tar-fixed-metadata-gzip-mtime-zero-v1'},
                        'files': files, 'excluded': excluded}
            snapshot_id = hashlib.sha256(_json_bytes(identity)).hexdigest()
            manifest = {**identity, 'snapshot_id': snapshot_id,
                        'complete_tasks': complete, 'total_tasks': len(tasks), 'tasks': tasks,
                        'source_bytes': packed_bytes, 'archive_bytes': writer.position,
                        'compression': 'concatenated-parts-of-one-deterministic-tar-gzip-stream',
                        'parts': writer.parts,
                        'scope': 'All planned task directories (including trial predictions, validation history and task checkpoints), source plan and existing aggregate report/CSV files.',
                        'not_included': ['raw EEG datasets', 'shared source EEGNet models outside this round output',
                                         'operational runtime/log directories', 'private credentials', 'symbolic links'],
                        'report_recomputed': False, 'source_modified': False,
                        'aggregate_report_capture': 'Existing report and CSV files are frozen independently per opened file descriptor; they can be asynchronous relative to completed task receipts. Later report replacements or newly generated tables do not invalidate this task-output snapshot.',
                        'whole_environment_backup_claimed': False}
            _atomic_json(stage / 'manifest.json', manifest)
            _atomic_json(stage / 'summary-files.json', summaries)
            destination = round_work / snapshot_id
            if destination.exists():
                # Exact content identity gives an identical deterministic archive;
                # replace a previous incomplete/corrupt local copy under our lock.
                shutil.rmtree(destination)
            os.replace(stage, destination)
            # Reuse only if summaries still match the versions actually
            # captured, and task files match their final verified signatures.
            # Never label a later report version as already archived.
            cache_inventory = dict(after_map)
            cache_inventory.update(captured_summary_sources)
            source_signature = hashlib.sha256(_json_bytes({
                'plan_sha256': hashlib.sha256(plan_data).hexdigest(),
                'completion_digest': completion_digest,
                'files': [cache_inventory[key] for key in sorted(cache_inventory)],
                'excluded': excluded, 'part_bytes': part_bytes})).hexdigest()
            _atomic_json(round_work / 'prepared.json', {
                'round': round_number, 'snapshot_id': snapshot_id,
                'manifest_sha256': hashlib.sha256(_json_bytes(manifest)).hexdigest(),
                'source_signature': source_signature, 'summary_files': summaries})
            _emit(emit, 'PACK_COMPLETE', round=round_number, snapshot_id=snapshot_id,
                  files=len(files), parts=len(writer.parts), archive_bytes=writer.position)
            return _ready_result(destination, manifest, summaries, False)
        finally:
            if writer.handle is not None:
                writer.handle.close()
