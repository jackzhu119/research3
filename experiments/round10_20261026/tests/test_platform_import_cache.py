import hashlib
import json
from pathlib import Path
import sys

from paper3r10.platform.import_cache import verified_cache_paths


def fixture(tmp_path):
    root=tmp_path/'project';raw={'python':sys.version,'executable':sys.executable}
    identity=hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    cache=tmp_path/'node'/identity;cache.mkdir(parents=True)
    for name in ('stdlib','site'):(cache/name).mkdir()
    (cache/'manifest.json').write_text(json.dumps(raw))
    receipt=root/'.paper3-hpc/jobs/769863/node_cache.json';receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({'status':'NODE_LOCAL_CACHE_VERIFIED','cache_root':str(cache),
                                  'manifest_sha256':identity,'python_executable':sys.executable}))
    return root,cache


def test_verified_owned_same_python_cache_adopted_without_walk(tmp_path,monkeypatch):
    root,cache=fixture(tmp_path)
    monkeypatch.setattr(Path,'rglob',lambda *_:(_ for _ in ()).throw(AssertionError('No recursive rehash')))
    assert verified_cache_paths(root)==[str(cache/'stdlib'),str(cache/'site')]


def test_invalid_or_other_python_cache_falls_back_without_gate(tmp_path):
    root,cache=fixture(tmp_path);(cache/'manifest.json').write_text('{}')
    assert verified_cache_paths(root)==[]
    assert verified_cache_paths(tmp_path/'absent')==[]
