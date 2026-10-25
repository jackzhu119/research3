"""Reconstruct the exact original report from its lossless GitHub JSON manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper3_result_transport import _transport_context


def request(url, *, tries=10):
    """Public result URLs only; no credentials in commands or output."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60, context=_transport_context()) as response:
                return response.read()
        except (OSError, TimeoutError, urllib.error.URLError):
            if attempt + 1 == tries:
                raise
            time.sleep(min(30, 2 ** attempt))


def restore(manifest, output, read_part):
    output = Path(output)
    temporary = output.with_name(output.name + '.part')
    digest, offset = hashlib.sha256(), 0
    with temporary.open('wb') as handle:
        for expected_number, part in enumerate(manifest['parts'], 1):
            if part['number'] != expected_number or part['offset_bytes'] != offset:
                raise ValueError('Report chunk order or byte offsets differ')
            block = read_part(part)
            if len(block) != part['bytes'] or hashlib.sha256(block).hexdigest() != part['sha256']:
                raise ValueError('Report chunk size or SHA256 differs')
            handle.write(block)
            digest.update(block)
            offset += len(block)
    if offset != manifest['source_bytes'] or digest.hexdigest() != manifest['source_sha256']:
        raise ValueError('Restored report full SHA256 differs')
    os.replace(temporary, output)
    return dict(bytes=offset, sha256=digest.hexdigest(), output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='Local manifest path or HTTPS raw.githubusercontent.com URL')
    parser.add_argument('--output', default='paper3-round7-report.json')
    parser.add_argument('--ref', default='paper3-results', help='Prefer receipt commit SHA for immutable retrieval')
    args = parser.parse_args()
    if args.manifest.startswith('https://'):
        manifest = json.loads(request(args.manifest))
        ref = urllib.parse.quote(args.ref, safe='')
        read = lambda part: request('https://raw.githubusercontent.com/jackzhu119/research3/' + ref + '/' + part['path'])
    else:
        source = Path(args.manifest)
        manifest = json.loads(source.read_bytes())
        read = lambda part: (source.parent / part['local_name']).read_bytes()
    print(json.dumps(dict(status='RESTORED_SHA256_VERIFIED', **restore(manifest, args.output, read))))


if __name__ == '__main__':
    main()
