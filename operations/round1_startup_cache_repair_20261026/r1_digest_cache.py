"""Reuse real file SHA256 reads across original-campaign action processes.

This operational cache changes no acceptance decision: callers still compare
the returned digest against their existing frozen evidence. Entries originate
only from successful original hash calls, with unchanged stat identity before
and after the read. Cache failure is a cold miss, never a training failure.
"""
from __future__ import annotations

from contextlib import contextmanager
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time

SCHEMA = "paper3-r1-real-file-sha256-cache-v1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_STAT = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")


def _signature(path):
    value = path.stat()
    if not stat.S_ISREG(value.st_mode):
        raise OSError("digest cache accepts regular files only")
    result = [getattr(value, key) for key in _STAT]
    if any(type(item) is not int for item in result):
        raise OSError("precise file identity unavailable")
    return result


def _source_sha(path):
    # Small implementation files only; never use a caller-declared digest.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DigestCache:
    def __init__(self, root, namespace, *, emit=None):
        self.root = Path(root).resolve()
        self.namespace = namespace
        self.path = self.root / "runs/training_campaign_parallel_v1/digest-cache-v1/index.json"
        self.emit = emit
        self.entries = self._load()
        self.dirty = {}
        self.saved = []
        self.mapping = {}
        self.stats = {"hits": 0, "misses": 0, "bytes_reused": 0,
                      "hashes_saved": 0, "cache_io_errors": 0}
        self.last_flush = time.monotonic()
        self.last_notice = self.last_flush

    def _load(self):
        try:
            if self.path.stat().st_size > 64 * 1024 * 1024:
                return {}
            document = json.loads(self.path.read_text())
            if (document.get("schema") == SCHEMA and document.get("namespace") == self.namespace
                    and isinstance(document.get("entries"), dict)):
                return document["entries"]
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        return {}

    def _eligible(self, path):
        try:
            canonical = Path(path).resolve(strict=True)
            canonical.relative_to(self.root)
            # Cache bookkeeping cannot participate in its own identity.
            if canonical == self.path or self.path.parent in canonical.parents:
                return None
            return canonical
        except (OSError, ValueError, TypeError, RuntimeError):
            return None

    def get(self, original, path, *, sized):
        canonical = self._eligible(path)
        if canonical is None:
            return original(path)
        try:
            before = _signature(canonical)
        except OSError:
            return original(path)
        key = str(canonical)
        row = self.entries.get(key)
        if (isinstance(row, dict) and row.get("stat") == before
                and isinstance(row.get("sha256"), str) and _HEX.fullmatch(row["sha256"])
                and row.get("bytes") == before[2]):
            try:
                stable = _signature(canonical) == before and self._eligible(path) == canonical
            except OSError:
                stable = False
            if stable:
                self.stats["hits"] += 1
                self.stats["bytes_reused"] += before[2]
                self.progress(canonical)
                return (row["sha256"], before[2]) if sized else row["sha256"]
        self.stats["misses"] += 1
        self.progress(canonical)
        result = original(path)  # Original failures retain their original semantics.
        digest = result[0] if sized and isinstance(result, tuple) and len(result) == 2 else result
        if not isinstance(digest, str) or not _HEX.fullmatch(digest):
            return result
        if sized and (not isinstance(result, tuple) or len(result) != 2 or result[1] != before[2]):
            return result
        try:
            stable = _signature(canonical) == before and self._eligible(path) == canonical
        except OSError:
            stable = False
        if stable:
            row = {"stat": before, "sha256": digest, "bytes": before[2]}
            self.entries[key] = row
            self.dirty[key] = row
            self.stats["hashes_saved"] += 1
        # Bound recovery loss for long cold checks without per-file NFS writes.
        if len(self.dirty) >= 64 and time.monotonic() - self.last_flush >= 300:
            self.flush()
        return result

    def progress(self, path):
        now = time.monotonic()
        if now - self.last_notice >= 30:
            self.last_notice = now
            self.notice("R1_FILE_DIGEST_CACHE_PROGRESS", file=path.name, **self.stats)

    def flush(self):
        if not self.dirty:
            return
        temporary = None
        try:
            import fcntl
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.with_suffix(".lock").open("a") as lock:
                # Three cards may finish together. Busy cache => recompute later.
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                merged = self._load()
                merged.update(self.dirty)
                descriptor, temporary = tempfile.mkstemp(prefix=".digest-", dir=self.path.parent)
                with os.fdopen(descriptor, "w") as stream:
                    json.dump({"schema": SCHEMA, "namespace": self.namespace, "entries": merged},
                              stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
                    stream.write("\n")
                os.replace(temporary, self.path)
                temporary = None
                self.dirty.clear()
                self.last_flush = time.monotonic()
        except (OSError, ValueError, TypeError, ImportError):
            self.stats["cache_io_errors"] += 1
        finally:
            if temporary:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def install(self, targets):
        for original, sized in targets:
            @functools.wraps(original)
            def wrapper(path, _original=original, _sized=sized):
                return self.get(_original, path, sized=_sized)
            wrapper._paper3_r1_digest_cache = self
            self.mapping[original] = wrapper
        self.refresh_aliases()

    def refresh_aliases(self):
        # Canonical definitions and already-imported aliases have the same
        # function identity. Later from-imports inherit the patched canonical.
        for name, module in list(sys.modules.items()):
            if not (name == "p3work" or name.startswith("p3work.")
                    or name == "tools" or name.startswith("tools.")):
                continue
            namespace = getattr(module, "__dict__", None)
            if not isinstance(namespace, dict):
                continue
            for key, value in list(namespace.items()):
                for original, wrapper in self.mapping.items():
                    if value is original:
                        setattr(module, key, wrapper)
                        self.saved.append((module, key, original, wrapper))
                        break

    def close(self):
        self.flush()
        # This also restores aliases imported after installation.
        reverse = {id(wrapper): original for original, wrapper in self.mapping.items()}
        for name, module in list(sys.modules.items()):
            if not (name == "p3work" or name.startswith("p3work.")
                    or name == "tools" or name.startswith("tools.")):
                continue
            namespace = getattr(module, "__dict__", None)
            if isinstance(namespace, dict):
                for key, value in list(namespace.items()):
                    original = reverse.get(id(value))
                    if original is not None:
                        setattr(module, key, original)
        self.notice("R1_FILE_DIGEST_CACHE_SUMMARY", **self.stats)

    def notice(self, event, **fields):
        try:
            if self.emit is None:
                print(json.dumps({"event": event,
                                  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                  **fields}, sort_keys=True), flush=True)
            else:
                self.emit(event, **fields)
        except Exception:
            pass


@contextmanager
def active_digest_cache(root, action=None, *, emit=None):
    """Patch hash reads only; run every existing validator and actual GPU probe."""
    cache = None
    try:
        contracts = importlib.import_module("p3work.contracts")
        backup = importlib.import_module("tools.r2_backup")
        namespace = {"schema": SCHEMA, "root": str(Path(root).resolve()),
                     "file_digest_source": _source_sha(contracts.__file__),
                     "sha256_file_source": _source_sha(backup.__file__)}
        cache = DigestCache(root, namespace, emit=emit)
        cache.install([(contracts.file_digest, False), (backup.sha256_file, True)])
        cache.notice("R1_FILE_DIGEST_CACHE_ACTIVE", loaded_entries=len(cache.entries),
                     source_hashes_reused_from_declarations=False)
    except (OSError, ValueError, TypeError, ImportError, AttributeError) as error:
        # This optimization must never create a new preflight blocker.
        if cache is not None:
            cache.close()
        cache = None
        try:
            if emit:
                emit("R1_FILE_DIGEST_CACHE_UNAVAILABLE", error_type=type(error).__name__)
        except Exception:
            pass
    try:
        yield cache
    finally:
        if cache is not None:
            cache.close()


def install_process_cache(root, *, emit=None):
    """Enable the same cache in fresh owned-stage children until normal exit.

    Multiprocessing finalizers run even though multiprocessing uses os._exit,
    while atexit covers ordinary Python children. This grants no work capability.
    """
    try:
        contracts = importlib.import_module("p3work.contracts")
    except (ImportError, AttributeError):
        return None
    existing = getattr(contracts.file_digest, "_paper3_r1_digest_cache", None)
    if existing is not None:
        return existing
    context = active_digest_cache(root, emit=emit)
    cache = context.__enter__()
    if cache is None:
        return None
    import atexit
    from multiprocessing.util import Finalize
    state = {"closed": False}
    def finish():
        if not state["closed"]:
            state["closed"] = True
            context.__exit__(None, None, None)
    atexit.register(finish)
    Finalize(None, finish, exitpriority=10)
    return cache
