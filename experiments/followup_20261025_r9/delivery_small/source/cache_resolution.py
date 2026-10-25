"""Resolve only the 114 declared public cache files from earlier local installs.

No training data tree traversal, NumPy import or source modification occurs.
The two original manifests are embedded byte-for-byte. X/events/schema files are
copied by their exact declared identities. Fallback files are each below1MiB.
"""
import hashlib
import json
import os
from pathlib import Path
import ssl
import sys
import time
import urllib.parse
import urllib.request
import zipfile

INSTALL_NAMES = ('paper3-round8', 'paper3-round7', 'paper3-round6-hpcgpu',
    'paper3-round6', 'paper3-round5-hpcgpu', 'paper3-round5', 'paper3-round4-hpcgpu',
    'paper3-round4', 'paper3-round3', 'paper3-followup', 'paper3-rounds-hpcgpu',
    'paper3-round9')


def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields), ensure_ascii=False), flush=True)


def signature(path):
    st = Path(path).stat()
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def requirements(package):
    rows = []
    for folder in ('pan2023', 'weibo'):
        manifest = json.loads((Path(package) / 'data' / folder / 'manifest.json').read_text())
        for session in manifest['sessions']:
            for field in ('X','events','schema'):
                item = session[field]
                relative = Path('data') / folder / item['path']
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Invalid declared cache path')
                rows.append(dict(path=relative.as_posix(), bytes=item['bytes'], sha256=item['sha256']))
    return rows


def candidate_roots(home, explicit=()):
    """Only list the direct children of known installation directories."""
    home = Path(home)
    found, seen = [], set()
    def add(path):
        path = Path(path).expanduser()
        if str(path) not in seen and path.is_dir():
            seen.add(str(path)); found.append(path)
    for path in explicit:
        add(path)
    for name in INSTALL_NAMES:
        base = home / name
        if base.is_dir():
            add(base)
            for child in sorted(base.iterdir(), reverse=True):
                if child.is_dir():
                    add(child)
                    # Historic wrappers sometimes put the experiment below a
                    # named payload directory; do not recursively search it.
                    for suffix in ('payload', 'round6', 'round7', 'round8'):
                        add(child / suffix)
    for name in ('followup_20261025_r8', 'followup_20261024_r7', 'followup_20261024_r6'):
        add(home / 'projects' / 'paper3' / 'experiments' / name)
    return found


def candidate_files(root, relative):
    relative = Path(relative)
    values = [Path(root) / relative]
    if Path(root).name == 'data':
        values.append(Path(root) / Path(*relative.parts[1:]))
    if Path(root).name == relative.parts[1]:
        values.append(Path(root) / Path(*relative.parts[2:]))
    return values


def exact_bytes(path, item):
    path = Path(path)
    try:
        before = signature(path)
        if before[2] != item['bytes']:
            return None
        with path.open('rb') as handle:
            data = handle.read()
            st = os.fstat(handle.fileno())
            after_fd = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
        if before != after_fd or signature(path) != before:
            return None
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            return None
        return data
    except OSError:
        return None


def place(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temp.open('wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def context():
    ctx = ssl.create_default_context()
    prefix = Path(sys.executable).absolute().parent.parent
    version = 'python%d.%d' % sys.version_info[:2]
    for folder in ('lib', 'lib64'):
        for name in ('botocore/cacert.pem', 'certifi/cacert.pem'):
            path = prefix / folder / version / 'site-packages' / name
            if path.is_file():
                ctx.load_verify_locations(cafile=str(path))
    return ctx


def download_one(item, bases, *, opener=None, timeout=15):
    opener = opener or urllib.request.urlopen
    for base in bases:
        url = base.rstrip('/') + '/' + urllib.parse.quote(item['path'], safe='/')
        try:
            request = urllib.request.Request(url, headers={
                'User-Agent': 'paper3-r9-small-array-reuse/1', 'Accept-Encoding': 'identity'})
            with opener(request, timeout=timeout, context=context()) as response:
                data = response.read(item['bytes'] + 1)
            if len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']:
                return data, url
            emit('R9_SMALL_ARRAY_RETRY', path=item['path'], reason='size_or_hash')
        except (OSError, ValueError) as error:
            emit('R9_SMALL_ARRAY_RETRY', path=item['path'], error_type=type(error).__name__)
    return None, None


def archive_cache(package, specification, *, opener=None):
    """Optional fallback only; normal prior-round cache reuse never calls this."""
    opener = opener or urllib.request.urlopen
    url, separator, fragment = specification.partition('#sha256=')
    if not separator or len(fragment) != 64 or any(c not in '0123456789abcdef' for c in fragment):
        raise ValueError('Fallback archive requires URL#sha256=<64 hex>')
    work = Path(package) / '.data-downloads'
    work.mkdir(exist_ok=True)
    target = work / (fragment + '.zip')
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == fragment:
        return target
    partial = target.with_suffix('.part')
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {'User-Agent':'paper3-r9-optional-cache/1', 'Accept-Encoding':'identity'}
    if offset:
        headers['Range'] = 'bytes=%d-' % offset
    request = urllib.request.Request(url, headers=headers)
    with opener(request, timeout=15, context=context()) as response:
        status = getattr(response, 'status', None)
        if offset and status == 206:
            if not response.headers.get('Content-Range','').startswith('bytes %d-' % offset):
                raise RuntimeError('Archive byte range does not match cached offset')
            mode = 'ab'
        else:
            offset = 0
            mode = 'wb'
        with partial.open(mode) as handle:
            while True:
                block = response.read(256*1024)
                if not block:
                    break
                handle.write(block)
                handle.flush()
                offset += len(block)
                emit('R9_OPTIONAL_CACHE_DOWNLOAD_PROGRESS', bytes=offset)
    if hashlib.sha256(partial.read_bytes()).hexdigest() != fragment:
        partial.unlink(missing_ok=True)
        raise RuntimeError('Fallback archive hash differs; retry a complete copy')
    os.replace(partial,target)
    return target


def ensure_data(package, *, cache_home=None, cache_roots=(), data_bases=(), data_archives=(),
                retry_seconds=5, max_cycles=None, opener=None):
    package = Path(package)
    home = Path(cache_home) if cache_home is not None else Path.home()
    rows = requirements(package)
    receipt_path = package / 'LOCAL_DATA_READY.json'
    try:
        prior = json.loads(receipt_path.read_text())
    except (OSError, ValueError):
        prior = {}
    prior_rows = {row['path']: row for row in prior.get('files', [])}
    roots = candidate_roots(home, cache_roots)
    missing, records = [], []
    emit('R9_LOCAL_CACHE_REUSE_STARTED', arrays=len(rows),
         known_install_roots=len(roots), task_tree_scan=False)
    for item in rows:
        target = package / item['path']
        old = prior_rows.get(item['path'], {})
        try:
            ready = old.get('sha256') == item['sha256'] and old.get('signature') == signature(target)
        except OSError:
            ready = False
        if ready:
            records.append(old)
            continue
        data = exact_bytes(target, item)
        origin = str(target) if data is not None else None
        if data is None:
            for root in roots:
                for candidate in candidate_files(root, item['path']):
                    data = exact_bytes(candidate, item)
                    if data is not None:
                        origin = str(candidate)
                        break
                if data is not None:
                    break
        if data is None:
            missing.append(item)
            continue
        if origin != str(target):
            place(target, data)
        records.append(dict(item, signature=signature(target), origin=origin))
        emit('R9_LOCAL_ARRAY_REUSED', path=item['path'], bytes=item['bytes'])
    if missing and not data_bases and not data_archives:
        emit('R9_DATA_CACHE_MISSING', missing_arrays=len(missing),
             paths=[row['path'] for row in missing],
             source_code_installed=True, school_jobs_submitted=False)
        raise RuntimeError('Some declared arrays were not found; supply --cache-root or --data-base. Existing training was not changed.')
    cycle = 0
    while missing:
        cycle += 1
        for specification in data_archives:
            if not missing:
                break
            try:
                archive = archive_cache(package, specification, opener=opener)
                remaining = []
                with zipfile.ZipFile(archive) as packed:
                    for item in missing:
                        try:
                            data = packed.read(item['path'])
                        except KeyError:
                            remaining.append(item)
                            continue
                        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                            remaining.append(item)
                            continue
                        target = package / item['path']
                        place(target,data)
                        records.append(dict(item,signature=signature(target),origin=specification))
                        emit('R9_DECLARED_CACHE_FILE_RESTORED',path=item['path'],bytes=item['bytes'])
                missing = remaining
            except Exception as error:
                emit('R9_OPTIONAL_CACHE_DOWNLOAD_RETRY',error_type=type(error).__name__,local_cache_preserved=True)
        remaining = []
        for item in missing:
            data, origin = download_one(item, data_bases, opener=opener)
            if data is None:
                remaining.append(item)
                continue
            target = package / item['path']
            place(target, data)
            records.append(dict(item, signature=signature(target), origin=origin))
            emit('R9_SMALL_ARRAY_DOWNLOADED', path=item['path'], bytes=item['bytes'])
        # Save successful individual arrays even if another endpoint is down.
        place(receipt_path, (json.dumps({'files':records}, sort_keys=True, indent=2)+'\n').encode())
        missing = remaining
        if missing:
            if max_cycles is not None and cycle >= max_cycles:
                raise RuntimeError('Array download remains incomplete; cached arrays retained')
            emit('R9_DATA_DOWNLOAD_RETRY', missing_arrays=len(missing), retry_seconds=retry_seconds)
            time.sleep(retry_seconds)
    receipt = dict(status='LOCAL_DATA_READY', arrays=len(rows),
        bytes=sum(row['bytes'] for row in rows), files=sorted(records, key=lambda row:row['path']),
        scientific_manifests_unchanged=True, all_prior_rounds_untouched=True)
    place(receipt_path, (json.dumps(receipt, sort_keys=True, indent=2)+'\n').encode())
    emit('R9_LOCAL_DATA_READY', arrays=len(rows), bytes=receipt['bytes'],
         downloaded_arrays=sum(str(row.get('origin','')).startswith('https://') for row in records))
    return receipt

