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
    if round_number not in (2, 3, 7):
        raise ValueError('This exporter supports rounds 2, 3 and 7 only')
    if not isinstance(part_bytes, int) or part_bytes < 1024:
        raise ValueError('Archive part size must be at least 1024 bytes')
    name = 'followup_20261023_v2' if round_number == 2 else 'followup_20261024_r%d' % round_number
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


"""Stdlib GitHub result transport; independent of training and reporting code."""
import base64
import hashlib
import json
import os
from pathlib import Path
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


class ApiError(Exception):
    def __init__(self, status, message="GitHub rejected the request"):
        self.status = status
        # Never propagate server response bodies, request headers or credentials.
        super().__init__("%s (HTTP %s)" % (message, status))


class _TransportRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise ApiError(code, "Non-HTTPS GitHub redirect refused")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urllib.parse.urlsplit(newurl).hostname not in (
                "api.github.com", "uploads.github.com"):
            for key in list(redirected.headers):
                if key.lower() == "authorization":
                    del redirected.headers[key]
            for key in list(redirected.unredirected_hdrs):
                if key.lower() == "authorization":
                    del redirected.unredirected_hdrs[key]
        return redirected


def _transport_context():
    context = ssl.create_default_context()
    prefixes = {Path(sys.executable).absolute().parent.parent,
                Path(sys.executable).resolve().parent.parent}
    candidates = []
    for prefix in prefixes:
        for directory in prefix.glob("lib/python*/site-packages"):
            candidates.extend((directory / "botocore/cacert.pem", directory / "certifi/cacert.pem"))
    # -S deliberately omits site-packages, but its CA bundle remains usable.
    for directory in sys.path:
        if directory:
            candidates.extend((Path(directory) / "botocore/cacert.pem",
                               Path(directory) / "certifi/cacert.pem"))
    for path in candidates:
        if path.is_file():
            context.load_verify_locations(cafile=str(path))
    return context


class GitHub:
    def __init__(self, token, emit, *, repo="jackzhu119/research3", branch="paper3-results",
                 context=None, timeout=60, cache_path=None):
        self._token = token
        self.emit = emit
        self.repo, self.branch = repo, branch
        self.prefix = "repos/" + repo
        self.timeout = timeout
        self.context = context or _transport_context()
        self.opener = urllib.request.build_opener(
            _TransportRedirect(), urllib.request.HTTPSHandler(context=self.context))
        self.cache_path = Path(cache_path) if cache_path else None
        try:
            self.blob_cache = json.loads(self.cache_path.read_text()) if self.cache_path else {}
            if not isinstance(self.blob_cache, dict):
                self.blob_cache = {}
        except (OSError, ValueError):
            self.blob_cache = {}

    def _emit(self, event, **fields):
        self.emit(event, **fields)

    def _delay(self, attempt, headers=None):
        delay = min(60, 2 ** min(attempt, 6))
        headers = headers or {}
        try:
            requested = float(headers.get("Retry-After", "0"))
            delay = min(60, max(delay, requested))
        except (TypeError, ValueError):
            pass
        self._emit("GITHUB_RETRY", attempt=attempt, delay_seconds=delay)
        time.sleep(delay)

    @staticmethod
    def _retryable(error, body=b""):
        if error.code in (408, 425, 429, 500, 502, 503, 504):
            return True
        if error.code == 403:
            return (error.headers.get("X-RateLimit-Remaining") == "0"
                    or bool(error.headers.get("Retry-After"))
                    or b"secondary rate limit" in body.lower()
                    or b"rate limit exceeded" in body.lower())
        return False

    def _request(self, url, data=None, method=None, *, accept="application/vnd.github+json",
                 content_type="application/json"):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in ("api.github.com", "uploads.github.com"):
            raise ValueError("Unsupported GitHub endpoint")
        request = urllib.request.Request(url, data=data, method=method,
            headers={"Authorization": "Bearer " + self._token,
                     "User-Agent": "paper3-results-archive-uploader",
                     "Accept": accept, "Content-Type": content_type,
                     "X-GitHub-Api-Version": "2022-11-28"})
        return self.opener.open(request, timeout=self.timeout)

    def api(self, path, body=None, method=None, allow404=False):
        url = path if path.startswith("https://") else "https://api.github.com/" + path.lstrip("/")
        encoded = None if body is None else json.dumps(body, allow_nan=False).encode()
        attempt = 0
        while True:
            try:
                with self._request(url, encoded, method) as response:
                    raw = response.read()
                return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as error:
                raw = error.read(65536)
                if allow404 and error.code == 404:
                    return None
                if not self._retryable(error, raw):
                    raise ApiError(error.code) from None
                attempt += 1
                self._delay(attempt, error.headers)
            except (OSError, urllib.error.URLError, TimeoutError) as error:
                attempt += 1
                self._emit("GITHUB_NETWORK_RETRY", error_type=type(error).__name__)
                self._delay(attempt)

    def ensure_branch(self):
        endpoint = self.prefix + "/git/ref/heads/" + urllib.parse.quote(self.branch, safe="/")
        found = self.api(endpoint, allow404=True)
        if found is not None:
            return found["object"]["sha"]
        default = self.api(self.prefix)["default_branch"]
        head = self.api(self.prefix + "/git/ref/heads/" + urllib.parse.quote(default, safe="/"))["object"]["sha"]
        try:
            return self.api(self.prefix + "/git/refs",
                {"ref": "refs/heads/" + self.branch, "sha": head}, "POST")["object"]["sha"]
        except ApiError as error:
            if error.status not in (409, 422):
                raise
            # Another uploader, or a lost creation acknowledgement, can create it.
            return self.api(endpoint)["object"]["sha"]

    def ensure_release(self, tag, title, body):
        endpoint = self.prefix + "/releases/tags/" + urllib.parse.quote(tag, safe="")
        found = self.api(endpoint, allow404=True)
        if found is not None:
            return found
        self.ensure_branch()
        try:
            return self.api(self.prefix + "/releases", {
                "tag_name": tag, "target_commitish": self.branch, "name": title,
                "body": body, "draft": False, "prerelease": False,
                "make_latest": "false"}, "POST")
        except ApiError as error:
            if error.status not in (409, 422):
                raise
            found = self.api(endpoint, allow404=True)
            if found is None:
                raise
            return found

    def _assets(self, release):
        assets, page = [], 1
        while True:
            batch = self.api(self.prefix + "/releases/%s/assets?per_page=100&page=%s" % (release["id"], page))
            assets.extend(batch)
            if len(batch) < 100:
                return assets
            page += 1

    def _asset_hash(self, asset):
        digest = asset.get("digest") or ""
        if digest.startswith("sha256:"):
            return digest[7:]
        attempt = 0
        while True:
            try:
                digest = hashlib.sha256()
                with self._request(self.prefix_url + "/releases/assets/%s" % asset["id"],
                                   accept="application/octet-stream") as response:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        digest.update(block)
                return digest.hexdigest()
            except urllib.error.HTTPError as error:
                raw = error.read(65536)
                if not self._retryable(error, raw):
                    raise ApiError(error.code, "Cannot verify uploaded archive part") from None
                attempt += 1
                self._delay(attempt, error.headers)
            except (OSError, urllib.error.URLError, TimeoutError) as error:
                attempt += 1
                self._emit("ASSET_VERIFY_RETRY", error_type=type(error).__name__)
                self._delay(attempt)

    @property
    def prefix_url(self):
        return "https://api.github.com/" + self.prefix

    def upload_part(self, release, path, name, sha256):
        path = Path(path)
        size = path.stat().st_size
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != sha256:
            raise ValueError("Local archive part hash differs from its manifest")
        upload_url = release["upload_url"].split("{", 1)[0]
        upload_url += "?" + urllib.parse.urlencode({"name": name})
        attempt = 0
        while True:
            # Always re-list before POST, including after a lost POST response.
            matches = [a for a in self._assets(release) if a.get("name") == name]
            for asset in matches:
                if (asset.get("state") == "uploaded" and asset.get("size") == size
                        and self._asset_hash(asset) == sha256):
                    self._emit("ARCHIVE_PART_VERIFIED", name=name, bytes=size, reused=True)
                    return {"id": asset["id"], "name": name, "bytes": size, "sha256": sha256,
                            "url": asset.get("browser_download_url"), "reused": True}
                # This asset was found in this exact release with this exact name.
                self.api(self.prefix + "/releases/assets/%s" % asset["id"], method="DELETE", allow404=True)
            try:
                self._emit("ARCHIVE_PART_UPLOADING", name=name, bytes=size)
                with self._request(upload_url, path.read_bytes(), "POST",
                                   content_type="application/octet-stream") as response:
                    created = json.load(response)
                if (created.get("state") != "uploaded" or created.get("size") != size
                        or self._asset_hash(created) != sha256):
                    self._emit("ARCHIVE_PART_VERIFY_RETRY", name=name)
                    attempt += 1
                    self._delay(attempt)
                    continue
                self._emit("ARCHIVE_PART_VERIFIED", name=name, bytes=size, reused=False)
                return {"id": created["id"], "name": name, "bytes": size, "sha256": sha256,
                        "url": created.get("browser_download_url"), "reused": False}
            except urllib.error.HTTPError as error:
                raw = error.read(65536)
                # 422 can acknowledge an already completed POST whose reply was lost.
                if not self._retryable(error, raw) and error.code not in (409, 422):
                    raise ApiError(error.code, "Archive part upload rejected") from None
                attempt += 1
                self._delay(attempt, error.headers)
            except (OSError, urllib.error.URLError, TimeoutError) as error:
                attempt += 1
                self._emit("ARCHIVE_PART_RETRY", name=name, error_type=type(error).__name__)
                self._delay(attempt)

    def _save_cache(self):
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.cache_path.with_name(self.cache_path.name + ".tmp.%s" % os.getpid())
        temp.write_text(json.dumps(self.blob_cache, sort_keys=True))
        os.replace(temp, self.cache_path)

    def _blob(self, content):
        if not isinstance(content, bytes):
            raise TypeError("Git content must be bytes")
        # Git SHA is deterministic and works across repository branches/reruns.
        expected = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if self.blob_cache.get(expected) == expected:
            return expected
        result = self.api(self.prefix + "/git/blobs", {
            "content": base64.b64encode(content).decode(), "encoding": "base64"}, "POST")
        if result.get("sha") != expected:
            raise ValueError("GitHub blob identity mismatch")
        self.blob_cache[expected] = expected
        self._save_cache()
        return expected

    def git_publish(self, files, message):
        self.ensure_branch()
        entries = []
        for path, content in sorted(files.items()):
            pure = Path(path)
            if pure.is_absolute() or ".." in pure.parts or not path:
                raise ValueError("Unsafe result repository path")
            entries.append({"path": path, "mode": "100644", "type": "blob", "sha": self._blob(content)})
        endpoint = self.prefix + "/git/ref/heads/" + urllib.parse.quote(self.branch, safe="/")
        attempt = 0
        while True:
            head = self.api(endpoint)["object"]["sha"]
            base_tree = self.api(self.prefix + "/git/commits/" + head)["tree"]["sha"]
            tree = self.api(self.prefix + "/git/trees", {"base_tree": base_tree, "tree": entries}, "POST")["sha"]
            if tree == base_tree:
                self._emit("RESULT_INDEX_UNCHANGED", branch=self.branch, commit=head)
                return head
            commit = self.api(self.prefix + "/git/commits", {
                "message": message, "tree": tree, "parents": [head]}, "POST")["sha"]
            try:
                self.api(self.prefix + "/git/refs/heads/" + urllib.parse.quote(self.branch, safe="/"),
                         {"sha": commit, "force": False}, "PATCH")
                self._emit("RESULT_INDEX_PUBLISHED", branch=self.branch, commit=commit)
                return commit
            except ApiError as error:
                if error.status not in (409, 422):
                    raise
                attempt += 1
                self._emit("RESULT_INDEX_REBASE_RETRY", branch=self.branch)
                self._delay(attempt)


"""Upload completed Paper3 rounds; never import an experiment or control Slurm."""
import argparse
import fcntl
import getpass
import glob
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

UPLOAD_REPOSITORY = 'jackzhu119/research3'
UPLOAD_BRANCH = 'paper3-results'


def upload_emit(event, **fields):
    print(json.dumps({'event': event, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      **fields}, ensure_ascii=False, sort_keys=True), flush=True)


def upload_json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def upload_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('wb') as handle:
        handle.write(upload_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def upload_read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def upload_discover_token(explicit=None):
    for name in ('PAPER3_GITHUB_TOKEN', 'GITHUB_TOKEN', 'GH_TOKEN'):
        token = os.environ.get(name, '').strip()
        if token:
            return token
    candidates = []
    for candidate in (explicit, os.environ.get('PAPER3_GITHUB_CREDENTIALS')):
        if candidate:
            candidates.append(Path(candidate).expanduser())
    candidates.append(Path.home()/'.config/paper3/github.json')
    for name in ('paper3-followup', 'paper3-round3', 'paper3-round4', 'paper3-round5',
                 'paper3-round6', 'paper3-round7', 'paper3-round4-hpcgpu',
                 'paper3-round5-hpcgpu', 'paper3-round6-hpcgpu'):
        base = Path.home()/name
        candidates.append(base/'private/github.json')
        candidates.extend(sorted(base.glob('*/private/github.json')))
    for path in candidates:
        value = upload_read(path, {})
        token = value.get('token') if isinstance(value, dict) else None
        if isinstance(token, str) and token.strip():
            return token.strip()
    raise RuntimeError('未找到现有GitHub凭证；可用 --credentials 指定原有 private/github.json。凭证不进入归档。')


def upload_remote_round(gh, packed, work):
    number = packed['round']
    manifest = packed['manifest']
    snapshot = packed['snapshot_id']
    prefix = 'results/paper3/round-%02d/final/%s' % (number, snapshot)
    tag = 'paper3-results-r%02d-%s' % (number, snapshot)
    title = 'Paper3 第%d轮完整输出 %s' % (number, snapshot)
    description = ('第%d轮 %d/%d 任务完成。全部计划任务的预测、验证轨迹、选择文件和checkpoint见分片归档。\n'
                   '按 part 编号拼接后为一个 tar.gz；每片及每个解压文件的SHA256见清单。\n'
                   '原始EEG、共享源EEGNet、安装环境及运行控制文件不属于本轮输出归档。\n'
                   '可浏览索引：https://github.com/%s/tree/%s/%s') % (
                       number, packed['complete_tasks'], packed['total_tasks'],
                       UPLOAD_REPOSITORY, UPLOAD_BRANCH, prefix)
    release = gh.ensure_release(tag, title, description)
    assets = []
    for index, part in enumerate(packed['parts'], 1):
        upload_emit('PART_UPLOADING', round=number, part=index, total_parts=len(packed['parts']),
                    bytes=part['bytes'], name=part['name'])
        record = gh.upload_part(release, Path(part['path']), part['name'], part['sha256'])
        assets.append({'name': part['name'], 'bytes': part['bytes'], 'sha256': part['sha256'],
                       'asset_id': record['id'], 'download_url': record['url']})
        upload_emit('PART_VERIFIED', round=number, part=index, total_parts=len(packed['parts']))
    # An identical immutable manifest is also present among the browsable Git files.
    files = {}
    omitted_browsable = []
    for record in packed.get('summary_files', []):
        relative = record.get('relative_path', record.get('relative', record.get('path_in_archive', record.get('path'))))
        local = record.get('local_path', record.get('path'))
        if not relative or not local:
            raise ValueError('Packager summary_files must provide relative_path and local_path')
        data = Path(local).read_bytes()
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Frozen summary checksum changed')
        if len(data) > 16*1024*1024:
            omitted_browsable.append(relative)
            continue
        files[prefix+'/files/'+relative] = data
    files[prefix+'/manifest.json'] = upload_json_bytes(manifest)
    files[prefix+'/TASKS.json'] = upload_json_bytes(manifest.get('tasks', []))
    archive = {
        'schema_version': 'paper3-complete-output-archive-v1', 'round': number,
        'snapshot_id': snapshot, 'status': 'ALL_PARTS_UPLOADED_AND_SHA256_VERIFIED',
        'complete_tasks': packed['complete_tasks'], 'total_tasks': packed['total_tasks'],
        'release_tag': tag, 'release_url': release['html_url'],
        'manifest_sha256': hashlib.sha256(files[prefix+'/manifest.json']).hexdigest(),
        'manifest_path': prefix+'/manifest.json', 'parts': assets,
        'part_order': 'Concatenate parts in this array order to reconstruct one tar.gz stream.',
        'existing_reports_recomputed': False,
        'large_browsable_files_available_in_archive': omitted_browsable,
        'scope': 'Complete declared-round scientific output; not raw EEG, shared source model or environment backup.',
    }
    files[prefix+'/archive.json'] = upload_json_bytes(archive)
    readme = ('# 第%d轮完整结果\n\n本次逐任务回执：**%d/%d 完成**。\n\n'
              '- [原计划与已有报告、CSV](files/)\n- [任务完成索引](TASKS.json)\n'
              '- [全部科学文件SHA256清单](manifest.json)\n- [完整分片下载索引](archive.json)\n'
              '- [GitHub Release：完整预测、验证轨迹与checkpoint](%s)\n\n'
              '分片按 archive.json 中的顺序拼接为 tar.gz；下载校验/还原工具见仓库 '
              '`operations/results-upload/RESTORE-PAPER3-RESULTS.py`。\n\n'
              '报告和CSV复制自服务器既有文件，不在上传时重新统计。'
              '若已有报告仍标PARTIAL，完整任务原始输出与本次完成索引均已包含，不能把旧报告计数当成缺文件。\n\n'
              '原始EEG、共享源模型、环境和运行控制文件不在本轮归档范围。\n') % (
                  number, packed['complete_tasks'], packed['total_tasks'], release['html_url'])
    files[prefix+'/README_ZH.md'] = readme.encode()
    latest = {'round': number, 'snapshot_id': snapshot,
              'status': 'COMPLETE_OUTPUT_UPLOADED', 'complete_tasks': packed['complete_tasks'],
              'total_tasks': packed['total_tasks'], 'archive': prefix+'/archive.json',
              'manifest': prefix+'/manifest.json', 'results': prefix+'/README_ZH.md',
              'release_url': release['html_url'],
              'verified_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    files['results/paper3/round-%02d/LATEST.json'%number] = upload_json_bytes(latest)
    files['results/paper3/round-%02d/README_ZH.md'%number] = (
        '# 第%d轮结果\n\n完整输出已上传：**%d/%d**。\n\n'
        '- [最新完整结果](final/%s/README_ZH.md)\n- [机器可读最新索引](LATEST.json)\n'
        '- [以前自动回传的快照](previous/)\n\n'
        '完整输出包括逐试次预测和checkpoint；以前监控回传只是异步汇总，两类分别保留。\n'
        ) % (number, packed['complete_tasks'], packed['total_tasks'], snapshot)
    round_readme = 'results/paper3/round-%02d/README_ZH.md'%number
    files[round_readme] = files[round_readme].encode()
    upload_emit('GIT_INDEX_UPLOADING', round=number, files=len(files))
    commit = gh.git_publish(files, 'Archive completed Paper3 round %d output %s'%(number, snapshot))
    latest['commit'] = commit
    upload_atomic(Path(work)/('round-%02d-UPLOADED.json'%number), latest)
    upload_emit('ROUND_UPLOAD_COMPLETE', round=number, complete_tasks=packed['complete_tasks'],
                total_tasks=packed['total_tasks'], commit=commit, release_url=release['html_url'])
    return latest


def upload_worker(args):
    work = Path(args.work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    lock = (work/'worker.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        upload_emit('UPLOAD_ALREADY_RUNNING', log=str(work/'upload.log'))
        return 0
    upload_atomic(work/'worker.json', {'pid': os.getpid(), 'started_utc': time.time(), 'rounds': args.rounds})
    try:
        os.nice(10)
    except OSError:
        pass
    upload_emit('UPLOAD_WORKER_STARTED', rounds=args.rounds, root=args.root, repository=UPLOAD_REPOSITORY,
                branch=UPLOAD_BRANCH, training_modified=False, log=str(work/'upload.log'))
    gh = GitHub(upload_discover_token(args.credentials), upload_emit,
                repo=UPLOAD_REPOSITORY, branch=UPLOAD_BRANCH, cache_path=work/'git-blob-cache.json')
    gh.ensure_branch()
    todo, completed = list(args.rounds), []
    while todo:
        for number in list(todo):
            try:
                upload_emit('ROUND_SCAN_STARTED', round=number)
                packed = prepare_round(Path(args.root), number, work, upload_emit)
                if packed['status'] != 'READY':
                    upload_emit('ROUND_NOT_COMPLETE_YET', round=number,
                                complete_tasks=packed.get('complete_tasks'), total_tasks=packed.get('total_tasks'))
                    continue
                previous = upload_read(work/('round-%02d-UPLOADED.json'%number), {})
                if previous.get('snapshot_id') == packed['snapshot_id']:
                    upload_emit('ROUND_PREVIOUS_UPLOAD_RECHECK', round=number)
                latest = upload_remote_round(gh, packed, work)
                completed.append(latest)
                todo.remove(number)
                upload_atomic(work/'status.json', {'status': 'UPLOADING' if todo else 'COMPLETE',
                    'completed_rounds': [r['round'] for r in completed], 'remaining_rounds': todo,
                    'completed': completed})
            except Exception as error:
                # Only error types are logged: exception messages may contain an HTTP body or credential.
                upload_emit('ROUND_UPLOAD_RETRY', round=number, error_type=type(error).__name__,
                            local_parts_preserved=True, other_training_unchanged=True)
                upload_atomic(work/'status.json', {'status': 'RETRYING', 'round': number,
                    'error_type': type(error).__name__, 'remaining_rounds': todo,
                    'completed_rounds': [r['round'] for r in completed]})
        if todo:
            upload_emit('RETRY_WAIT', seconds=60, remaining_rounds=todo)
            time.sleep(60)
    upload_emit('ALL_REQUESTED_ROUNDS_UPLOADED', rounds=args.rounds,
                index_url='https://github.com/'+UPLOAD_REPOSITORY+'/tree/'+UPLOAD_BRANCH+'/results/paper3')
    return 0


def upload_start(args):
    work = Path(args.work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    with (work/'launch.lock').open('a') as launch:
        fcntl.flock(launch, fcntl.LOCK_EX)
        with (work/'worker.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                upload_emit('UPLOAD_ALREADY_RUNNING', log=str(work/'upload.log'),
                            view_command='tail -n 40 '+str(work/'upload.log'))
                return 0
            fcntl.flock(lock, fcntl.LOCK_UN)
        # Confirm the existing credential is discoverable before detaching; never write its contents.
        upload_discover_token(args.credentials)
        command = [sys.executable, '-S', '-B', '-u', str(Path(__file__).resolve()), '--worker',
                   '--root', args.root, '--work-dir', str(work), '--rounds'] + list(map(str,args.rounds))
        if args.credentials:
            command += ['--credentials', args.credentials]
        with (work/'upload.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     start_new_session=True, close_fds=True)
        upload_emit('UPLOAD_BACKGROUND_STARTED', pid=child.pid, log=str(work/'upload.log'),
                    status=str(work/'status.json'), rounds=args.rounds,
                    view_command='tail -n 40 '+str(work/'upload.log'),
                    training_jobs_started_or_stopped=0)
    return 0


def upload_main():
    parser = argparse.ArgumentParser(description='Archive completed Paper3 rounds 2/3/7 without running experiments.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--start', action='store_true', help='Detach uploader and print its log path.')
    mode.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--root', default='/public/home/zhuziyuan/projects/paper3')
    parser.add_argument('--work-dir', default=str(Path.home()/'paper3-results-upload'))
    parser.add_argument('--rounds', nargs='+', type=int, choices=(2,3,7), default=[2,3,7])
    parser.add_argument('--credentials')
    args = parser.parse_args()
    if not args.start and not args.worker:
        parser.print_help()
        return 0
    return upload_start(args) if args.start else upload_worker(args)


if __name__ == '__main__':
    raise SystemExit(upload_main())
