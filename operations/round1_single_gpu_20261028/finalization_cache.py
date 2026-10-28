"""Reuse successful native validation inside one R1 finalization action.

No result is trusted from a declaration or a previous process. The native
verifier performs its complete first validation; only identical later calls in
this action reuse a private copy. Every local file read by that validation is
bound to an inode/size/mtime/ctime snapshot, including nested verifier reads.
The original immutable result writer runs only after a complete dependency
check. This module neither fits a model nor changes a scientific result.
"""
from __future__ import annotations

import builtins
from contextlib import contextmanager
import copy
import functools
import importlib
import inspect
import io
import json
import os
from pathlib import Path
import stat
import sys
import time

VERSION = 'paper3-r1-finalization-verification-reuse-v1'
SUPPORTED_KINDS = frozenset(('strongest_control', 'freeze'))
STATUS = {'strongest_control': 'STRONGEST_DEVELOPMENT_CONTROL_FROZEN',
          'freeze': 'IMMUTABLE_PRE_INTERNAL_DEVELOPMENT_FREEZE'}
STATE = '_paper3_r1_finalization_verification_reuse_v1'


class FinalizationDependencyChanged(ValueError):
    pass


def _signature(path):
    try:
        item = path.stat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(item.st_mode):
        raise FinalizationDependencyChanged('FINALIZATION_READ_NOT_REGULAR: ' + str(path))
    return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns,
            str(path.resolve(strict=True)))


def _plain(value):
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _plain(child) for key, child in value.items()}
    raise TypeError('FINALIZATION_VALIDATOR_ARGUMENT_NOT_JSON')


def _key(function, args, kwargs):
    bound = inspect.signature(function).bind(*args, **kwargs)
    bound.apply_defaults()
    return json.dumps(_plain(bound.arguments), sort_keys=True, separators=(',', ':'),
                      allow_nan=False)


class FinalizationSnapshot:
    def __init__(self, root, action, source_snapshot, development, *, emit=None):
        self.root = Path(root).resolve(strict=True)
        self.action, self.source, self.development = action, source_snapshot, development
        self.emit = emit
        target = action.get('expected_receipt')
        if not isinstance(target, str) or not target:
            raise FinalizationDependencyChanged('FINALIZATION_EXPECTED_RECEIPT_REQUIRED')
        self.target = (self.root / target).resolve(strict=False)
        if not self.target.is_relative_to(self.root / 'runs'):
            raise FinalizationDependencyChanged('FINALIZATION_RECEIPT_OUTSIDE_RUNS')
        self.dependencies = {}
        self.collectors = []
        self.entries = {}
        self.patches = []
        self.stats = dict(native_verifications=0, reused_verifications=0,
                          dependency_checks=0, publications_guarded=0)
        self.last_notice = time.monotonic()

    def notice(self, event, **extra):
        record = dict(version=VERSION, action_key=self.action.get('key'),
                      action_kind=self.action.get('kind'),
                      utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      **self.stats, dependencies=len(self.dependencies), **extra)
        try:
            if self.emit is None:
                print(json.dumps(dict(event=event, **record), ensure_ascii=False), flush=True)
            else:
                self.emit(event, **record)
        except Exception:
            pass

    def _path(self, value):
        if isinstance(value, int):
            return None  # File descriptors are not scientific artifact paths.
        try:
            path = Path(os.fsdecode(os.fspath(value))).absolute()
        except (TypeError, ValueError):
            return None
        # Libraries imported during validation live outside the project and are
        # not scientific input artifacts. Native path gates still own root and
        # symlink validation for every declared scientific reference.
        if not path.is_relative_to(self.root):
            return None
        actual = path.resolve(strict=False)
        if not actual.is_relative_to(self.root):
            raise FinalizationDependencyChanged('FINALIZATION_READ_ESCAPED_ROOT')
        return path

    def record_read(self, value):
        if not self.collectors:
            return
        path = self._path(value)
        if path is None:
            return
        signature = _signature(path)
        if path in self.dependencies and self.dependencies[path] != signature:
            raise FinalizationDependencyChanged('FINALIZATION_INPUT_CHANGED: ' + str(path))
        self.dependencies[path] = signature
        for collector in self.collectors:
            collector[path] = signature

    def check(self, dependencies):
        for path, signature in dependencies.items():
            if (not path.resolve(strict=False).is_relative_to(self.root)
                    or _signature(path) != signature):
                raise FinalizationDependencyChanged('FINALIZATION_INPUT_CHANGED: ' + str(path))
        self.stats['dependency_checks'] += 1

    def checkpoint(self):
        self.check(self.dependencies)
        self.source.checkpoint()

    def _patch(self, module, name, replacement):
        previous = getattr(module, name)
        setattr(module, name, replacement)
        self.patches.append((module, name, previous, replacement))

    def _replace_aliases(self, original, replacement):
        for name, module in list(sys.modules.items()):
            if module is None or not name.startswith(('p3work.', 'tools.')):
                continue
            for attribute, value in list(vars(module).items()):
                if value is original:
                    self._patch(module, attribute, replacement)

    def install(self):
        # Both Path.open/read_text (io.open) and NumPy/torch path loaders
        # (builtins.open) are observed. Reads are captured only while a native
        # verifier is executing; unrelated logging/heartbeat writes are ignored.
        for module in (builtins, io):
            original_open = module.open

            def make_open(original):
                @functools.wraps(original)
                def opened(file, mode='r', *args, **kwargs):
                    if isinstance(mode, str) and ('r' in mode or '+' in mode):
                        self.record_read(file)
                    return original(file, mode, *args, **kwargs)
                return opened

            self._patch(module, 'open', make_open(original_open))

        for name in ('verify_configuration_receipt', 'verify_development_receipt'):
            original = getattr(self.development, name)
            successes = {}
            self.entries[name] = successes

            def make_verifier(function, label, cache):
                @functools.wraps(function)
                def verified(*args, **kwargs):
                    bound = inspect.signature(function).bind(*args, **kwargs)
                    bound.apply_defaults()
                    if Path(bound.arguments['root']).resolve(strict=True) != self.root:
                        raise FinalizationDependencyChanged('FINALIZATION_VALIDATION_ROOT_DIFFERS')
                    identity = _key(function, args, kwargs)
                    if identity in cache:
                        result, dependencies = cache[identity]
                        self.check(dependencies)
                        # A cached nested verification contributes its complete
                        # transitive read graph to any outer verification.
                        for collector in self.collectors:
                            collector.update(dependencies)
                        self.stats['reused_verifications'] += 1
                        if time.monotonic() - self.last_notice >= 30:
                            self.last_notice = time.monotonic()
                            self.notice('R1_FINALIZATION_VERIFICATION_REUSED', verifier=label)
                        return copy.deepcopy(result)
                    reads = {}
                    self.collectors.append(reads)
                    try:
                        result = function(*args, **kwargs)
                        self.check(reads)
                        # Production receipt verifiers return only JSON values.
                        # Keep a private copy so callers cannot mutate a seal.
                        sealed = copy.deepcopy(result)
                    finally:
                        self.collectors.pop()
                    cache[identity] = (sealed, reads)
                    self.stats['native_verifications'] += 1
                    self.notice('R1_FINALIZATION_NATIVE_VERIFICATION_FINISHED', verifier=label,
                                verified_read_dependencies=len(reads))
                    return result
                return verified

            replacement = make_verifier(original, name, successes)
            self._replace_aliases(original, replacement)
            if getattr(self.development, name) is original:
                self._patch(self.development, name, replacement)

        original_put = self.development._put

        @functools.wraps(original_put)
        def guarded_put(path, value):
            actual = Path(path).resolve(strict=False)
            status = value.get('status') if isinstance(value, dict) else None
            if actual == self.target or status in STATUS.values():
                if actual != self.target or status != STATUS[self.action['kind']]:
                    raise FinalizationDependencyChanged('FINALIZATION_PUBLICATION_IDENTITY_DIFFERS')
                self.checkpoint()
                self.notice('R1_FINALIZATION_DEPENDENCIES_CHECKED_BEFORE_PUBLICATION',
                            original_immutable_writer_retained=True)
                result = original_put(path, value)
                self.stats['publications_guarded'] += 1
                return result
            return original_put(path, value)

        self._patch(self.development, '_put', guarded_put)
        self.notice('R1_FINALIZATION_VERIFICATION_REUSE_ACTIVE',
                    process_local=True, original_native_verifiers_retained=True,
                    source_cache_persists_across_actions=False,
                    original_scientific_files_changed=False)
        return self

    def close(self, *, successful):
        try:
            if successful:
                self.checkpoint()
        finally:
            for module, name, previous, replacement in reversed(self.patches):
                if getattr(module, name) is replacement:
                    setattr(module, name, previous)
            self.patches.clear()
            for cache in self.entries.values():
                cache.clear()
            self.notice('R1_FINALIZATION_VERIFICATION_REUSE_FINISHED', successful=successful,
                        source_cache_persists_across_actions=False)


@contextmanager
def active(root, action, old_verified_cache, *, training=None, development=None, emit=None):
    if action.get('kind') not in SUPPORTED_KINDS:
        yield None
        return
    training = training or importlib.import_module('p3work.classical_training')
    development = development or importlib.import_module('p3work.development_evaluation')
    source = old_verified_cache.ActionSnapshot(root, action, emit=emit).install(training, development)
    snapshot = None
    successful = False
    try:
        snapshot = FinalizationSnapshot(root, action, source, development, emit=emit)
        snapshot.install()
        yield snapshot
        successful = True
    finally:
        try:
            if snapshot is not None:
                snapshot.close(successful=successful)
        finally:
            source.close(successful=successful)


def install(startup_module, old_verified_cache, root, *, training=None, development=None, emit=None):
    expected_root = Path(root).resolve(strict=True)
    prior = getattr(startup_module, STATE, None)
    if prior is not None:
        if prior['root'] != expected_root or startup_module.active is not prior['wrapper']:
            raise FinalizationDependencyChanged('FINALIZATION_STARTUP_SCOPE_DIFFERS')
        return dict(status='ALREADY_INSTALLED', supported_kinds=sorted(SUPPORTED_KINDS), version=VERSION)
    previous = startup_module.active

    @contextmanager
    @functools.wraps(previous)
    def wrapped(root, action, *args, **kwargs):
        if action.get('kind') in SUPPORTED_KINDS and Path(root).resolve(strict=True) != expected_root:
            raise FinalizationDependencyChanged('FINALIZATION_STARTUP_ROOT_DIFFERS')
        selected_emit = emit if emit is not None else kwargs.get('emit')
        with active(root, action, old_verified_cache, training=training,
                    development=development, emit=selected_emit):
            with previous(root, action, *args, **kwargs) as result:
                yield result

    startup_module.active = wrapped
    setattr(startup_module, STATE, dict(root=expected_root, original=previous, wrapper=wrapped))
    return dict(status='R1_FINALIZATION_REUSE_INSTALLED', version=VERSION,
                supported_kinds=sorted(SUPPORTED_KINDS), scientific_files_changed=False,
                existing_children_modified=False)


def restore(startup_module):
    prior = getattr(startup_module, STATE, None)
    if prior is None or startup_module.active is not prior['wrapper']:
        return False
    startup_module.active = prior['original']
    delattr(startup_module, STATE)
    return True
