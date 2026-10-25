"""Activate an already verified node cache, then execute the original entry.

No package installation, directory copying or environment tree hashing occurs.
The original worker remains the sole owner of scientific results/checkpoints.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def emit(event, **fields):
    print(json.dumps({'event':event,**fields},sort_keys=True),flush=True)


def verified_cache_paths(root, executable=None, version=None):
    executable = executable or sys.executable
    version = version or sys.version
    root = Path(root)
    for job in ('769863','770043'):
        try:
            receipt = json.loads((root/'.paper3-hpc/jobs'/job/'node_cache.json').read_text())
            cache = Path(receipt['cache_root'])
            if (receipt.get('status')!='NODE_LOCAL_CACHE_VERIFIED'
                    or receipt.get('python_executable')!=executable
                    or not cache.is_absolute() or cache.is_symlink()
                    or cache.name!=receipt.get('manifest_sha256') or cache.stat().st_uid!=os.getuid()):
                continue
            raw = json.loads((cache/'manifest.json').read_text())
            digest=hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            paths=[cache/'stdlib',cache/'site']
            if (digest!=receipt['manifest_sha256'] or raw.get('python')!=version
                    or raw.get('executable')!=executable
                    or any(not p.is_dir() or p.is_symlink() for p in paths)):
                continue
            return [str(p) for p in paths]
        except (OSError,ValueError,KeyError,TypeError):
            continue
    return []


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if len(args)<3 or args[0]!='--original-entry':
        raise SystemExit('Usage: backend_entry.py --original-entry ORIGINAL_ENTRY worker --plan PATH ...')
    entry=Path(args[1]).resolve()
    rest=args[2:]
    if entry.name!='entry.py' or entry.parent.name not in ('paper3r9',) or not entry.is_file():
        raise ValueError('Expected the existing Round9 scientific entry.py')
    plan_path=Path(rest[rest.index('--plan')+1])
    plan=json.loads(plan_path.read_text())
    cache=verified_cache_paths(plan['root'])
    if cache:
        sys.path[:0]=[p for p in cache if p not in sys.path]
        emit('VERIFIED_NODE_RUNTIME_REUSED',cache_paths=cache,runtime_files_rehashed=0,files_copied=0)
    else:
        emit('SHARED_RUNTIME_SELECTED',runtime_files_rehashed=0,files_copied=0,
             note='Original entry reports import progress every 30 seconds.')
    sys.argv=[str(entry),*rest]
    runpy.run_path(str(entry),run_name='__main__')


if __name__=='__main__':main()
