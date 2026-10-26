from __future__ import annotations

import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, "/workspace/paper3_hpc")
import r1_digest_cache as helper
from p3work import contracts
from tools import r2_backup


def _spawn_digest(root, expected, connection):
    from p3work import contracts as native
    # Both children execute the same original native function. Counting opens
    # demonstrates second-child cached reads, rather than testing a mock hash.
    original_open = Path.open
    reads = []
    target = Path(root) / "source.bin"
    def counted(path, *args, **kwargs):
        if Path(path) == target and args and args[0] == "rb":
            reads.append(str(path))
        return original_open(path, *args, **kwargs)
    Path.open = counted
    try:
        with helper.active_digest_cache(root, emit=lambda *a, **k: None) as cache:
            actual = native.file_digest(target)
            connection.send({"digest": actual, "expected": expected, "reads": len(reads),
                             "hits": cache.stats["hits"]})
    finally:
        Path.open = original_open
        connection.close()


def _spawn_process_cache(root, connection):
    target = Path(root) / "source.bin"
    first = helper.install_process_cache(root, emit=lambda *a, **k: None)
    assert helper.install_process_cache(root) is first
    connection.send(contracts.file_digest(target))
    connection.close()


def test_real_hash_alias_types_mutations_and_lazy_import(tmp_path):
    target = tmp_path / "source.bin"
    target.write_bytes(b"first")
    expected = hashlib.sha256(b"first").hexdigest()
    alias = types.ModuleType("p3work._cache_test_alias")
    alias.hash_read = contracts.file_digest
    sys.modules[alias.__name__] = alias
    original = contracts.file_digest
    try:
        with helper.active_digest_cache(tmp_path, emit=lambda *a, **k: None) as cache:
            assert alias.hash_read(target) == expected
            assert r2_backup.sha256_file(target) == (expected, 5)
            assert cache.stats["misses"] == 1 and cache.stats["hits"] == 1
            lazy = types.ModuleType("tools._cache_test_lazy")
            lazy.sha256_file = r2_backup.sha256_file
            sys.modules[lazy.__name__] = lazy
            assert lazy.sha256_file(target) == (expected, 5)
            before = target.stat()
            target.write_bytes(b"other")
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
            changed = hashlib.sha256(b"other").hexdigest()
            assert contracts.file_digest(target) == changed
            assert cache.stats["misses"] == 2
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"third")
            os.replace(replacement, target)
            assert alias.hash_read(target) == hashlib.sha256(b"third").hexdigest()
            assert cache.stats["misses"] == 3
        assert alias.hash_read is original and contracts.file_digest is original
        assert lazy.sha256_file is r2_backup.sha256_file
    finally:
        sys.modules.pop(alias.__name__, None)
        sys.modules.pop("tools._cache_test_lazy", None)


def test_cache_reused_by_real_spawned_native_hash(tmp_path):
    data = b"real immutable source input" * 1000
    (tmp_path / "source.bin").write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    context = multiprocessing.get_context("spawn")
    observations = []
    for _ in range(2):
        parent, child = context.Pipe(False)
        process = context.Process(target=_spawn_digest, args=(str(tmp_path), expected, child))
        process.start()
        child.close()
        assert parent.poll(15)
        observations.append(parent.recv())
        process.join(15)
        assert process.exitcode == 0
    assert [o["digest"] for o in observations] == [expected, expected]
    assert [o["reads"] for o in observations] == [1, 0]
    assert observations[1]["hits"] == 1


def test_hash_mutation_during_read_not_cached(tmp_path):
    target = tmp_path / "input"
    target.write_bytes(b"old")
    cache = helper.DigestCache(tmp_path, {"test": 1})
    def changing(path):
        result = hashlib.sha256(path.read_bytes()).hexdigest()
        path.write_bytes(b"new")
        return result
    assert cache.get(changing, target, sized=False) == hashlib.sha256(b"old").hexdigest()
    assert cache.entries == {} and cache.dirty == {}


def test_corrupt_cache_write_failure_and_outside_root_are_cold(tmp_path, monkeypatch):
    target = tmp_path / "input"
    target.write_bytes(b"value")
    namespace = {"test": 1}
    cache = helper.DigestCache(tmp_path, namespace)
    cache.path.parent.mkdir(parents=True)
    cache.path.write_text("broken json")
    cache = helper.DigestCache(tmp_path, namespace)
    assert cache.entries == {}
    digest = contracts.file_digest(target)
    assert cache.get(contracts.file_digest, target, sized=False) == digest
    monkeypatch.setattr(os, "replace", lambda *args: (_ for _ in ()).throw(OSError("read-only")))
    cache.flush()
    assert cache.stats["cache_io_errors"] == 1
    assert cache.get(contracts.file_digest, target, sized=False) == digest
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.write_bytes(b"outside")
    link = tmp_path / "escape"
    link.symlink_to(outside)
    assert cache.get(contracts.file_digest, link, sized=False) == contracts.file_digest(outside)
    assert str(outside) not in cache.entries


def test_original_validator_still_checks_expected_sha(tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"proof")
    def validator(expected):
        if contracts.file_digest(path) != expected:
            raise ValueError("input SHA differs")
    with helper.active_digest_cache(tmp_path, emit=lambda *a, **k: None):
        validator(hashlib.sha256(b"proof").hexdigest())
        with pytest.raises(ValueError, match="input SHA differs"):
            validator("0" * 64)


def test_cache_never_records_failed_or_wrong_sized_hash(tmp_path):
    target = tmp_path / "input"
    target.write_bytes(b"content")
    cache = helper.DigestCache(tmp_path, {"test": 1})
    def failed(path):
        raise OSError("real read failure")
    with pytest.raises(OSError, match="real read failure"):
        cache.get(failed, target, sized=False)
    assert cache.get(lambda path: ("a" * 64, 999), target, sized=True) == ("a" * 64, 999)
    assert not cache.entries


def test_symlink_retarget_between_hit_stats_rehashes_new_target(tmp_path, monkeypatch):
    first, second, link = (tmp_path / name for name in ("first", "second", "link"))
    first.write_bytes(b"first")
    second.write_bytes(b"other")
    link.symlink_to(first)
    cache = helper.DigestCache(tmp_path, {"test": 1})
    cache.get(contracts.file_digest, link, sized=False)
    original = helper._signature
    calls = []
    def retarget(path):
        value = original(path)
        calls.append(path)
        if len(calls) == 2:
            link.unlink()
            link.symlink_to(second)
        return value
    monkeypatch.setattr(helper, "_signature", retarget)
    assert cache.get(contracts.file_digest, link, sized=False) == hashlib.sha256(b"other").hexdigest()
    assert cache.stats["hits"] == 0


def test_spawned_owned_child_process_cache_flushes_on_exit(tmp_path):
    data = b"source fit metadata"
    (tmp_path / "source.bin").write_bytes(data)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(False)
    process = context.Process(target=_spawn_process_cache, args=(str(tmp_path), child))
    process.start()
    child.close()
    assert parent.poll(15)
    assert parent.recv() == hashlib.sha256(data).hexdigest()
    process.join(15)
    assert process.exitcode == 0
    with helper.active_digest_cache(tmp_path, emit=lambda *a, **k: None) as cache:
        assert contracts.file_digest(tmp_path / "source.bin") == hashlib.sha256(data).hexdigest()
        assert cache.stats["hits"] == 1


def test_independent_cache_snapshots_merge_after_other_process_flush(tmp_path):
    first = helper.DigestCache(tmp_path, {"test": 1})
    second = helper.DigestCache(tmp_path, {"test": 1})
    for cache, name in ((first, "a"), (second, "b")):
        target = tmp_path / name
        target.write_bytes(name.encode())
        cache.get(contracts.file_digest, target, sized=False)
    first.flush()
    second.flush()
    merged = helper.DigestCache(tmp_path, {"test": 1})
    assert {Path(key).name for key in merged.entries} == {"a", "b"}
