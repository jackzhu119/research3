"""Optional reuse of already verified, owned school node runtime caches.

No runtime copy, import test, recursive walk or fallback training gate occurs.
The previous deployment verified the complete cache; these small receipts bind it
to this exact executable and Python build, and any mismatch uses normal paths.
"""
import hashlib
import json
import os
from pathlib import Path
import sys


def verified_cache_paths(root, executable=None, version=None):
    executable=executable or sys.executable;version=version or sys.version
    root=Path(root)
    for job in ('769863','770043'):
        try:
            p=root/'.paper3-hpc/jobs'/job/'node_cache.json'
            if p.is_symlink() or p.stat().st_size>65536:continue
            receipt=json.loads(p.read_text());cache=Path(receipt['cache_root'])
            if (receipt.get('status')!='NODE_LOCAL_CACHE_VERIFIED'
                    or receipt.get('python_executable')!=executable
                    or not cache.is_absolute() or cache.is_symlink()
                    or cache.name!=receipt.get('manifest_sha256') or cache.stat().st_uid!=os.getuid()):continue
            manifest=cache/'manifest.json'
            if manifest.is_symlink() or manifest.stat().st_size>1024*1024:continue
            raw=json.loads(manifest.read_text())
            identity=hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            paths=[cache/'stdlib',cache/'site']
            if (identity!=receipt['manifest_sha256'] or raw.get('python')!=version
                    or raw.get('executable')!=executable
                    or any(not p.is_dir() or p.is_symlink() or p.stat().st_uid!=os.getuid() for p in paths)):continue
            return [str(p) for p in paths]
        except (OSError,ValueError,KeyError,TypeError):continue
    return []
