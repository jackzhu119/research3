"""Independent, bounded-memory M12 paper and full-scientific GitHub delivery.

Git data blobs use the API's base64 encoding for arbitrary bytes. Binary files
are never decoded as UTF-8. Network/monitor/report failures only update their
own sidecar state; scientific results, task leases and workers are untouched.
"""
from __future__ import annotations

import argparse
import base64
from collections import OrderedDict
import gzip
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import ssl
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

from .common import atomic_bytes, atomic_json, best_effort_json, digest, emit, encoded, load_plan, lock, read_json, safe_read, sha_file, utc
from . import reports

PREFIX = 'results/paper3/round-12'
CSV_PART_BYTES = 4*1024*1024
BINARY_PART_BYTES = 8*1024*1024
_FORBIDDEN = {'.git', '.ssh', 'private', 'secrets', 'credentials', '__pycache__'}


class TransportError(Exception):
    def __init__(self, status=None):
        self.status = status
        # Do not carry response bodies, headers, request URLs or credentials.
        super().__init__('Independent result transport failed')


def discover_token(explicit=None):
    for name in ('PAPER3_GITHUB_TOKEN', 'GITHUB_TOKEN', 'GH_TOKEN'):
        value = os.environ.get(name, '').strip()
        if value:
            return value
    candidates = []
    for name in (explicit, os.environ.get('PAPER3_GITHUB_CREDENTIALS')):
        if name:
            candidates.append(Path(name).expanduser())
    candidates.append(Path.home()/'.config/paper3/github.json')
    for name in ('paper3-followup', 'paper3-round3', 'paper3-round4', 'paper3-round5',
                 'paper3-round6', 'paper3-round7', 'paper3-round9', 'paper3-round10',
                 'paper3-round11', 'paper3-round4-hpcgpu', 'paper3-round5-hpcgpu', 'paper3-round6-hpcgpu'):
        directory = Path.home()/name
        candidates.append(directory/'private/github.json')
        if directory.is_dir():
            candidates.extend(directory.glob('*/private/github.json'))
    for path in candidates:
        try:
            value = read_json(path, {}, max_bytes=64*1024)
        except (OSError, ValueError):
            continue
        token = value.get('token', value.get('github_token')) if isinstance(value, dict) else None
        if isinstance(token, str) and token.strip():
            return token.strip()
    raise TransportError()


def _ssl_context():
    context = ssl.create_default_context()
    prefix = Path(os.path.abspath(os.sys.executable)).parent.parent
    for folder in [*prefix.glob('lib/python*/site-packages'), *[Path(value) for value in os.sys.path if value]]:
        for name in ('botocore/cacert.pem', 'certifi/cacert.pem'):
            path = folder/name
            if path.is_file():
                context.load_verify_locations(cafile=str(path))
    return context


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Git-data JSON endpoints do not require redirects. Never forward auth.
        raise TransportError(code)


class GitHubClient:
    """Bounded attempts, binary-safe blobs, complete remote SHA verification."""
    def __init__(self, token, *, repo='jackzhu119/research3', branch='paper3-results',
                 request=None, cache_path=None, timeout=60):
        self._token = token
        self.repo = repo;self.branch = branch;self.timeout = timeout
        self.prefix = 'repos/'+repo
        self.opener = urllib.request.build_opener(_NoRedirect(), urllib.request.HTTPSHandler(context=_ssl_context()))
        self._request_override = request
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = OrderedDict()
        if self.cache_path and self.cache_path.is_file():
            # Append-only small proofs: stream the log rather than allocate a
            # campaign-sized JSON object on each upload or restart.
            with self.cache_path.open() as handle:
                for line in handle:
                    try: item = json.loads(line)
                    except ValueError: continue
                    if isinstance(item, dict) and isinstance(item.get('sha256'), str):
                        self.cache[item['sha256']] = item
                        while len(self.cache) > 8192: self.cache.popitem(last=False)

    def api(self, path, body=None, method=None, allow404=False):
        if self._request_override is not None:
            return self._request_override(path, body, method, allow404=allow404)
        if path.startswith('https://'):
            parsed = urllib.parse.urlsplit(path)
            if parsed.hostname != 'api.github.com':
                raise TransportError()
            url = path
        else:
            url = 'https://api.github.com/'+path.lstrip('/')
        data = None if body is None else json.dumps(body, allow_nan=False, separators=(',', ':')).encode()
        request = urllib.request.Request(url, data=data, method=method, headers={
            'Authorization': 'Bearer '+self._token, 'Accept': 'application/vnd.github+json',
            'Content-Type': 'application/json', 'User-Agent': 'paper3-m12-independent-upload',
            'X-GitHub-Api-Version': '2022-11-28'})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if allow404 and error.code == 404:
                return None
            raise TransportError(error.code) from None
        except (OSError, TimeoutError, urllib.error.URLError, ValueError):
            raise TransportError() from None

    def ensure_branch(self):
        path = self.prefix+'/git/ref/heads/'+urllib.parse.quote(self.branch, safe='/')
        value = self.api(path, allow404=True)
        if value is not None:
            return value['object']['sha']
        default = self.api(self.prefix)['default_branch']
        source = self.api(self.prefix+'/git/ref/heads/'+urllib.parse.quote(default, safe='/'))['object']['sha']
        try:
            return self.api(self.prefix+'/git/refs', {'ref':'refs/heads/'+self.branch, 'sha':source}, 'POST')['object']['sha']
        except TransportError as error:
            if error.status not in (409, 422):
                raise
            return self.api(path)['object']['sha']

    def blob(self, data):
        if not isinstance(data, bytes):
            raise TypeError('Git blob payload must be exact bytes')
        checksum = hashlib.sha256(data).hexdigest()
        cached = self.cache.get(checksum)
        if isinstance(cached, dict) and cached.get('bytes') == len(data) and cached.get('remote_GET_SHA256_verified'):
            return cached['git_blob_sha']
        result = self.api(self.prefix+'/git/blobs',
                          {'content':base64.b64encode(data).decode('ascii'), 'encoding':'base64'}, 'POST')
        blob_sha = result['sha']
        remote = self.api(self.prefix+'/git/blobs/'+blob_sha)
        if remote.get('encoding') != 'base64':
            raise TransportError()
        try:
            recovered = base64.b64decode(''.join(remote['content'].split()), validate=True)
        except (ValueError, KeyError):
            raise TransportError() from None
        if len(recovered) != len(data) or hashlib.sha256(recovered).hexdigest() != checksum:
            raise TransportError()
        record = dict(sha256=checksum, git_blob_sha=blob_sha, bytes=len(data), remote_GET_SHA256_verified=True)
        self.cache[checksum] = record
        while len(self.cache) > 8192: self.cache.popitem(last=False)
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open('ab') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                handle.write(json.dumps(record, separators=(',', ':')).encode()+b'\n')
                handle.flush();os.fsync(handle.fileno())
        return blob_sha

    def publish(self, files, message):
        """Compare-and-swap branch updates preserve every other round's tree."""
        if not files:
            return self.ensure_branch()
        for attempt in range(5):
            head = self.ensure_branch()
            commit = self.api(self.prefix+'/git/commits/'+head)
            entries = [dict(path=name, mode='100644', type='blob', sha=value) for name, value in sorted(files.items())]
            tree = self.api(self.prefix+'/git/trees', {'base_tree':commit['tree']['sha'], 'tree':entries}, 'POST')['sha']
            created = self.api(self.prefix+'/git/commits', {'message':message, 'tree':tree, 'parents':[head]}, 'POST')['sha']
            try:
                self.api(self.prefix+'/git/refs/heads/'+urllib.parse.quote(self.branch, safe='/'),
                         {'sha':created, 'force':False}, 'PATCH')
                observed = self.api(self.prefix+'/git/ref/heads/'+urllib.parse.quote(self.branch, safe='/'))['object']['sha']
                if observed != created:
                    # Another independent uploader may append a preserving commit.
                    self.api(self.prefix+'/git/commits/'+created)
                return created
            except TransportError as error:
                if error.status not in (409, 422) or attempt == 4:
                    raise
        raise TransportError()


def _safe_output_file(plan, path):
    root = Path(plan['output']).resolve();path = Path(path)
    relative = path.relative_to(root)
    if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root)
            or any(part in _FORBIDDEN or part.lower().startswith('.env') for part in relative.parts)
            or path.name.lower() in ('github.json', 'credentials.json')):
        raise ValueError('Undeclared/private archive source refused')
    return relative.as_posix()


def _file_signature(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _csv_records(handle):
    """Yield complete raw CSV records, preserving quoted newlines and CRLF."""
    buffer = bytearray();quoted = False
    for block in iter(lambda:handle.read(1024*1024), b''):
        for value in block:
            buffer.append(value)
            if value == 34:
                quoted = not quoted  # Doubled quotes toggle twice.
            if value == 10 and not quoted:
                yield bytes(buffer);buffer.clear()
    if quoted:
        raise ValueError('Unclosed CSV quoting in scientific output')
    if buffer:
        yield bytes(buffer)


def csv_parts(path, part_bytes=CSV_PART_BYTES):
    """Independently readable parts; strip repeated headers to restore exact CSV."""
    with Path(path).open('rb') as handle:
        records = iter(_csv_records(handle))
        header = next(records, b'')
        if not header:
            yield b'', dict(header_bytes=0, data_records=0)
            return
        batch = bytearray(header);count = 0
        for record in records:
            if count and len(batch)+len(record) > part_bytes:
                yield bytes(batch), dict(header_bytes=len(header), data_records=count)
                batch = bytearray(header);count = 0
            batch.extend(record);count += 1
        yield bytes(batch), dict(header_bytes=len(header), data_records=count)


def upload_file(client, remote_prefix, path, *, csv_chunk_bytes=CSV_PART_BYTES,
                binary_chunk_bytes=BINARY_PART_BYTES):
    """Return tree entries and a lossless original-byte reconstruction manifest."""
    path = Path(path);before = _file_signature(path);source_sha = sha_file(path)
    entries = {};parts = []
    if path.suffix.lower() == '.csv':
        iterator = csv_parts(path, csv_chunk_bytes)
        representation = 'csv-raw-record-parts-repeated-original-header'
    else:
        def chunks():
            with path.open('rb') as handle:
                while True:
                    data = handle.read(binary_chunk_bytes)
                    if not data:
                        break
                    yield data, {}
            if before[2] == 0:
                yield b'', {}
        iterator = chunks();representation = 'exact-byte-parts'
    reconstructed = hashlib.sha256();reconstructed_bytes = 0
    for index, (data, metadata) in enumerate(iterator):
        name = remote_prefix+'/part-%06d.bin'%index
        # CSV parts stay visibly named CSV for analysis readers.
        if path.suffix.lower() == '.csv':
            name = name[:-4]+'.csv'
        blob = client.blob(data);entries[name] = blob
        original = data if index == 0 or representation == 'exact-byte-parts' else data[metadata['header_bytes']:]
        reconstructed.update(original);reconstructed_bytes += len(original)
        parts.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                          git_blob_sha=blob, **metadata))
    if _file_signature(path) != before or reconstructed_bytes != before[2] or reconstructed.hexdigest() != source_sha:
        raise ValueError('Scientific source changed or lossless reconstruction differed')
    manifest = dict(original_name=path.name, original_bytes=before[2], original_sha256=source_sha,
                    representation=representation, parts=parts,
                    restore=('Concatenate first CSV part then later parts after removing exactly header_bytes; '
                             'otherwise concatenate all exact byte parts; verify original byte count and SHA256'),
                    remote_complete_blob_GET_SHA256_verified=True)
    raw = encoded(manifest)
    name = remote_prefix+'/manifest.json';entries[name] = client.blob(raw)
    return entries, manifest


def _task_sources(plan, task, result, channel):
    folder = Path(plan['output'])/'tasks'/task['key']
    values = OrderedDict()
    result_path = folder/'result.json'
    values[_safe_output_file(plan, result_path)] = result_path
    for item in reports._descriptors(result):
        path = reports.declared_path(plan, item)
        if channel == 'full' or path.name in ('metrics.csv', 'participant_summary.json',
                'mechanism_diagnostics.json', 'mechanism_diagnostics.csv', 'SCOPE.json'):
            values[_safe_output_file(plan, path)] = path
    return list(values.values())


def make_task_archive(plan, task, result, destination):
    """Stream deterministic complete declared artifacts into an immutable archive."""
    destination = Path(destination);destination.parent.mkdir(parents=True, exist_ok=True)
    sources = _task_sources(plan, task, result, 'full')
    signatures = {str(path):_file_signature(path) for path in sources}
    fd, temporary = tempfile.mkstemp(prefix='.archive-', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as compressed:
                with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                    for path in sources:
                        name = _safe_output_file(plan, path)
                        info = archive.gettarinfo(str(path), arcname=name)
                        info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
                        with path.open('rb') as handle:
                            archive.addfile(info, handle)
            raw.flush();os.fsync(raw.fileno())
        if any(_file_signature(path) != signatures[str(path)] for path in sources):
            raise ValueError('Scientific source changed during independent full archive')
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def _public_value(value):
    """Defense in depth for public scientific metadata, never credential stores."""
    if isinstance(value, dict):
        def public_key(key):
            key = str(key).lower()
            return not (key in ('token', 'registry', 'authorization', 'password', 'secret')
                        or any(word in key for word in ('credential', 'github_token', 'access_token', 'private_key'))
                        or key.endswith(('_token', '_password', '_secret')))
        return {key:_public_value(item) for key,item in value.items() if public_key(key)}
    if isinstance(value, (list, tuple)):
        return [_public_value(item) for item in value]
    return value


def _public_native_inputs(inputs):
    if not isinstance(inputs, dict):
        return {}
    allowed = ('schema_version', 'input_identity', 'manifest_descriptor', 'registry_origin',
               'source_by_seed', 'dictionary_by_seed', 'internal_data_identity', 'people',
               'reference_rule', 'reference_query_overlap')
    # The complete R11 registry is intentionally not copied into public plan
    # provenance. Frozen source/dictionary locators and hashes are sufficient.
    return _public_value({key:inputs[key] for key in allowed if key in inputs})


def _public_plan(plan):
    allowed = ('schema_version', 'round', 'experiment', 'recipe_sha256', 'root', 'output', 'config',
               'tasks', 'resources', 'input_identity', 'code_manifest', 'phase_task_counts',
               'scientific_scope', 'independent_confirmation', 'independent_data_status',
               'participants', 'unavailable_external')
    public = _public_value({key:plan[key] for key in allowed if key in plan})
    if isinstance(plan.get('native_inputs'), dict):
        public['native_inputs'] = _public_native_inputs(plan['native_inputs'])
    return public


def _state_path(plan, channel):
    return Path(plan.get('runtime', Path(plan['output'])/'runtime'))/('upload-'+channel+'.json')


def _publish_progress(plan, paper_state=None, full_state=None, *, client_factory=None):
    from .runtime import status
    snapshot = status(plan)
    snapshot['uploads'] = {name:{key:value for key,value in state.items()
                                 if key not in ('tasks','task_manifests')}
                           for name, state in [('paper',paper_state),('full',full_state)] if isinstance(state, dict)}
    snapshot['updated_utc'] = utc();snapshot['scientific_completion_claimed_by_upload'] = False
    factory = client_factory or (lambda branch:GitHubClient(discover_token(plan.get('github_credentials')),
                                                           repo=plan.get('github_repo','jackzhu119/research3'), branch=branch))
    client = factory('paper3-r12-progress')
    path = 'monitoring/paper3-round12/status.json'
    return client.publish({path:client.blob(encoded(snapshot))}, 'Update M12 mechanism experiment progress')


def run_once(plan, channel, *, client=None, publish_progress=True):
    """One independent upload attempt; any failure is resumable and non-scientific."""
    if channel not in ('paper', 'full'):
        raise ValueError('Independent channel must be paper or full')
    work = Path(plan.get('runtime', Path(plan['output'])/'runtime'))/'upload-work'/channel
    work.mkdir(parents=True, exist_ok=True)
    state_path = _state_path(plan, channel)
    state = safe_read(state_path, {}) or {}
    if state.get('recipe_sha256') not in (None, plan.get('recipe_sha256')):
        state = {}
    previous_tasks = state.get('tasks', {}) if isinstance(state.get('tasks', {}), dict) else {}
    try:
        client = client or GitHubClient(discover_token(plan.get('github_credentials')),
            repo=plan.get('github_repo','jackzhu119/research3'), branch='paper3-results',
            cache_path=work/'verified-blobs.jsonl')
        prefix = PREFIX+'/studies/'+str(plan.get('recipe_sha256','unidentified'))[:16]
        entries = {};tasks = dict(previous_tasks)
        total_declared_bytes = sum(int(row.get('original_bytes',0)) for row in tasks.values())
        for task, result in reports.completed_results(plan):
            result_path = Path(plan['output'])/'tasks'/task['key']/'result.json'
            result_sha = sha_file(result_path)
            if tasks.get(task['key'], {}).get('result_sha256') == result_sha:
                continue
            if channel == 'full':
                archive_path = work/'archives'/(digest(task['key'])[:24]+'-'+result_sha[:16]+'.tar.gz')
                archive_receipt = safe_read(archive_path.with_suffix('.sha256.json'), {}) or {}
                if (not archive_path.is_file() or archive_receipt.get('result_sha256') != result_sha
                        or archive_receipt.get('bytes') != archive_path.stat().st_size
                        or archive_receipt.get('sha256') != sha_file(archive_path)):
                    make_task_archive(plan, task, result, archive_path)
                    atomic_json(archive_path.with_suffix('.sha256.json'), dict(result_sha256=result_sha,
                        bytes=archive_path.stat().st_size, sha256=sha_file(archive_path)))
                remote = prefix+'/full/tasks/'+task['key']+'/'+result_sha[:16]
                staged, manifest = upload_file(client, remote, archive_path)
                count = 1
                size = archive_path.stat().st_size
            else:
                staged = {};manifests=[];size=0
                sources = _task_sources(plan, task, result, 'paper')
                for path in sources:
                    relative = _safe_output_file(plan,path)
                    remote = prefix+'/paper/tasks/'+task['key']+'/'+result_sha[:16]+'/'+relative
                    values, manifest = upload_file(client, remote, path)
                    staged.update(values);manifests.append(manifest);size+=manifest['original_bytes']
                count=len(sources)
                manifest=dict(files=manifests, original_bytes=size)
            entries.update(staged)
            task_manifest_path = prefix+'/'+channel+'/tasks/'+task['key']+'/'+result_sha[:16]+'/task-manifest.json'
            task_manifest = dict(task_key=task['key'], task_identity=digest(task),
                result_sha256=result_sha, channel=channel, original_bytes=size,
                content=manifest, remote_GET_SHA256_verified=True)
            manifest_raw = encoded(task_manifest)
            entries[task_manifest_path] = client.blob(manifest_raw)
            task_record = dict(result_sha256=result_sha, original_bytes=size, files=count,
                manifest_path=task_manifest_path, manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
                remote_GET_SHA256_verified=True)
            # Commit one completed task at a time so a later network failure
            # cannot discard all progress from a long-running channel.
            commit = client.publish(entries, 'Archive M12 '+channel+' completed task '+task['key'])
            entries.clear()
            tasks[task['key']] = task_record
            state = dict(schema_version='paper3-m12-upload-state-v1', channel=channel,
                         recipe_sha256=plan.get('recipe_sha256'), status='PARTIAL',
                         complete_tasks=len(tasks), total_tasks=len(plan.get('tasks',[])), tasks=tasks,
                         updated_utc=utc(), commit=commit, training_controlled=False,
                         original_scientific_files_modified=False)
            atomic_json(state_path, state)
        # Reports/config are incremental paper evidence, independent of archives.
        if channel == 'paper':
            for path in sorted((Path(plan['output'])/'reports').rglob('*')):
                if path.is_file() and path.suffix.lower() in ('.json','.csv','.md') and not any(part.startswith('.') for part in path.relative_to(Path(plan['output'])).parts):
                    remote = prefix+'/paper/'+_safe_output_file(plan,path)
                    values,_ = upload_file(client,remote,path);entries.update(values)
            for name in ('frozen_tau.json','frozen-inputs.json','pilot-decision.json','development-freeze.json'):
                path = Path(plan['output'])/name
                if path.is_file():
                    values,_ = upload_file(client,prefix+'/paper/'+name,path);entries.update(values)
            entries[prefix+'/paper/plan.json'] = client.blob(encoded(_public_plan(plan)))
        expected = {task['key'] for task in plan.get('tasks',[])}
        complete = bool(expected) and expected <= set(tasks)
        # Scientific STOP can intentionally leave gated later tasks unrun.
        # Keep PARTIAL coverage rather than claiming every declared artifact exists.
        pointer = dict(schema_version='paper3-m12-channel-pointer-v1', round=12, channel=channel,
                       recipe_sha256=plan.get('recipe_sha256'), status='COMPLETE' if complete else 'PARTIAL',
                       complete_tasks=len(tasks), total_tasks=len(expected),
                       original_bytes=sum(int(row.get('original_bytes',0)) for row in tasks.values()),
                       completed_task_keys_sha256=digest(sorted(tasks)), updated_utc=utc(),
                       remote_complete_blob_GET_SHA256_verified=True,
                       training_controlled=False, scientific_completion_claimed=False,
                       independent_from_other_channel=True,
                       task_manifests={key:dict(path=value.get('manifest_path'), sha256=value.get('manifest_sha256'))
                                       for key,value in tasks.items()})
        entries[prefix+'/'+channel.upper()+'_LATEST.json'] = client.blob(encoded(pointer))
        commit = client.publish(entries,'Update M12 '+channel+' evidence pointer')
        state = dict(pointer, commit=commit, tasks=tasks)
        atomic_json(state_path,state)
        if publish_progress:
            other = safe_read(_state_path(plan,'full' if channel=='paper' else 'paper'), {}) or {}
            try:
                _publish_progress(plan, state if channel=='paper' else other,
                                  state if channel=='full' else other)
            except Exception as error:
                emit('M12_PROGRESS_UPLOAD_RETRY',error_type=type(error).__name__,training_controlled=False)
        return state
    except Exception as error:
        failure = dict(schema_version='paper3-m12-upload-state-v1', channel=channel,
                       recipe_sha256=plan.get('recipe_sha256'), status='RETRYING', updated_utc=utc(),
                       complete_tasks=len(state.get('tasks', previous_tasks)),
                       tasks=state.get('tasks', previous_tasks), error_type=type(error).__name__,
                       http_status=getattr(error,'status',None), training_controlled=False,
                       scientific_files_changed=False)
        best_effort_json(state_path,failure)
        emit('M12_UPLOAD_RETRY', channel=channel, error_type=type(error).__name__,
             http_status=getattr(error,'status',None), training_controlled=False)
        return failure


def report_once(plan):
    path = Path(plan.get('runtime',Path(plan['output'])/'runtime'))/'reporter.json'
    try:
        # Scientific completion keys are cheap and stable; no new fit means no
        # repeated bootstrap/re-read of all old session tables.
        keys = sorted((task['key'], result.get('metrics_csv',{}).get('sha256')
                       if isinstance(result.get('metrics_csv'),dict) else None)
                      for task,result in reports.completed_results(plan))
        identity = digest(keys)
        prior = safe_read(path,{}) or {}
        if prior.get('inputs_sha256') == identity and (Path(plan['output'])/'reports/report.json').is_file():
            waiting = dict(prior,status='WAITING_FOR_NEW_COMPLETION',updated_utc=utc(),training_controlled=False)
            best_effort_json(path,waiting)
            return waiting
        report = reports.build_report(plan)
        state = dict(status=report['status'],complete_tasks=report['complete_paired_tasks'],
                     inputs_sha256=identity,updated_utc=utc(),training_controlled=False)
        best_effort_json(path,state)
        return state
    except Exception as error:
        state=dict(status='RETRYING',error_type=type(error).__name__,updated_utc=utc(),training_controlled=False)
        best_effort_json(path,state);emit('M12_REPORT_RETRY',error_type=type(error).__name__,training_controlled=False)
        return state


def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('command',choices=('report-loop','upload','status-loop'))
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--channel',choices=('paper','full'))
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--interval',type=float,default=120.)
    args=parser.parse_args(argv)
    _,plan=load_plan(args.plan)
    if args.command=='upload' and args.channel is None:
        parser.error('upload requires --channel')
    kind=args.command+('-'+args.channel if args.channel else '')
    runtime=Path(plan.get('runtime',Path(plan['output'])/'runtime'));runtime.mkdir(parents=True,exist_ok=True)
    with lock(runtime/(kind+'.lock'),blocking=False):
        while True:
            if args.command=='report-loop':value=report_once(plan)
            elif args.command=='upload':value=run_once(plan,args.channel)
            else:
                try:value=dict(status='PROGRESS_PUBLISHED',commit=_publish_progress(plan,
                    safe_read(_state_path(plan,'paper'),{}),safe_read(_state_path(plan,'full'),{})))
                except Exception as error:value=dict(status='RETRYING',error_type=type(error).__name__,training_controlled=False)
            if args.once:
                emit('M12_SIDECAR_OBSERVATION',command=args.command,channel=args.channel,
                     status=value.get('status'),training_controlled=False)
                return 0
            # Background publication never supplies a stop event to fitting.
            time.sleep(max(5.,args.interval))


if __name__=='__main__':
    raise SystemExit(main())
