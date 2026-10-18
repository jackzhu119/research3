#!/usr/bin/env python3
"""List or cache a pinned public Figshare release without reading EEG/labels.

The bundled HS-Stroke v6 metadata was fetched from the official API on
2026-10-18. Default is a local dry run. --execute downloads only within the
explicit byte budget; it does not extract archives, import data or train.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path


def digest(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2 ** 20), b''):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def validate_spec(item):
    name = item['name']
    if Path(name).name != name or '/' in name or '\\' in name or name in ('.', '..'):
        raise ValueError('Unsafe official filename')
    url = item['download_url']
    if not url.startswith('https://'):
        raise ValueError('HTTPS official download required')
    expected_md5 = item.get('computed_md5')
    if not expected_md5 or len(expected_md5) != 32:
        raise ValueError('Official MD5 is required')
    if not isinstance(item['size'], int) or item['size'] < 0:
        raise ValueError('Invalid declared size')


def cached_record(item, path):
    if path.stat().st_size != item['size']:
        raise ValueError(f'Existing file has wrong size; retained unchanged: {path}')
    md5, sha = digest(path)
    if md5 != item['computed_md5']:
        raise ValueError(f'Existing file has wrong MD5; retained unchanged: {path}')
    return {'file_id': item['id'], 'name': item['name'], 'size_bytes': item['size'],
            'md5': md5, 'sha256': sha, 'status': 'verified'}


def fetch_one(item, directory):
    validate_spec(item)
    path = directory / item['name']
    part = directory / (item['name'] + '.part')
    if path.exists():
        return cached_record(item, path)
    offset = part.stat().st_size if part.exists() else 0
    if offset > item['size']:
        raise ValueError(f'Partial file too large; retained for inspection: {part}')
    if offset < item['size']:
        headers = {'User-Agent': 'EEG-research-cache/1.0'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        request = urllib.request.Request(item['download_url'], headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            if offset and response.status == 206:
                content_range = response.headers.get('Content-Range', '')
                if not content_range.startswith(f'bytes {offset}-'):
                    raise ValueError('Resume offset not confirmed by server')
                mode = 'ab'
            elif response.status == 200:
                # A server that ignores Range sends a full object; restart the
                # temporary file, never append a full response to a partial one.
                mode, offset = 'wb', 0
            else:
                raise ValueError(f'Unexpected HTTP status: {response.status}')
            with part.open(mode) as stream:
                count = offset
                for chunk in iter(lambda: response.read(2 ** 20), b''):
                    count += len(chunk)
                    if count > item['size']:
                        raise ValueError('Response exceeds pinned file size')
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
    record = cached_record(item, part)
    os.replace(part, path)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'catalog' /
                        'hs_stroke_v6_official_metadata.json')
    parser.add_argument('--root', type=Path, default=Path('/workspace/paper3_data/sealed/hs_stroke_v6'))
    parser.add_argument('--only-name', action='append', default=[])
    parser.add_argument('--max-total-gib', type=float, default=8.0)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    manifest_bytes = args.manifest.read_bytes()
    data = json.loads(manifest_bytes)
    files = data['files']
    if args.only_name:
        selected = set(args.only_name)
        names = {f['name'] for f in files}
        if selected - names:
            parser.error('Requested filename absent from official release')
        files = [f for f in files if f['name'] in selected]
    for item in files:
        validate_spec(item)
    total = sum(f['size'] for f in files)
    if args.max_total_gib <= 0 or total > args.max_total_gib * 2 ** 30:
        parser.error(f'Selected total {total / 2**30:.3f} GiB exceeds budget')
    print(json.dumps({'title': data['title'], 'version': data['version'], 'doi': data['doi'],
                      'license': data['license']['name'], 'selected_files': len(files),
                      'selected_bytes': total, 'selected_gib': round(total / 2**30, 4),
                      'action': 'download_archives_only' if args.execute else 'dry_run',
                      'destination': str(args.root),
                      'role': 'SEALED_EXTERNAL_EVALUATION_NOT_A_TRAINING_SOURCE'}, ensure_ascii=False))
    for item in files:
        print(f"{item['id']}\t{item['size']}\t{item['name']}")
    if not args.execute:
        return 0
    args.root.mkdir(parents=True, exist_ok=True)
    # One process writes this release. A crashed process releases its OS lock.
    import fcntl
    with (args.root / '.download.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        remaining = sum(max(0, f['size'] - ((args.root / f['name']).stat().st_size
                                           if (args.root / f['name']).exists()
                                           else (args.root / (f['name'] + '.part')).stat().st_size
                                           if (args.root / (f['name'] + '.part')).exists() else 0))
                        for f in files)
        if shutil.disk_usage(args.root).free < remaining + 256 * 2**20:
            raise RuntimeError('Insufficient disk for pinned selection and 256 MiB reserve')
        receipt = {'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                   'doi': data['doi'], 'version': data['version'],
                   'raw_archives_only': True, 'eeg_read': False, 'training_started': False,
                   'files': [], 'started_unix': time.time()}
        receipt_path = args.root / 'download_receipt.json'
        for item in files:
            record = fetch_one(item, args.root)
            receipt['files'].append(record)
            receipt['updated_unix'] = time.time()
            tmp = receipt_path.with_suffix('.json.part')
            tmp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            os.replace(tmp, receipt_path)
            print(f"VERIFIED {item['name']}", flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f'FAILED: {error}', file=sys.stderr)
        sys.exit(1)
